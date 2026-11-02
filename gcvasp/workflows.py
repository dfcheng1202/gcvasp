"""ASE drives the ions; each calculator evaluation equilibrates the electrons."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

from .bias import HarmonicBias
from .io import check_identity, dumps, prepare_atoms, read_restart, snapshot, write_frames


def _start(atoms, directory, resume):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = atoms.copy()
    prepare_atoms(reference)
    checkpoint = directory / "latest.extxyz"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("Use resume=True or choose a new directory: " + str(directory))
        atoms = read_restart(checkpoint)
        check_identity(atoms, reference)
    else:
        atoms = reference
    return atoms, directory, checkpoint.exists()


def _cleanup_wavecars(calculators, keep_wavecar):
    if not keep_wavecar:
        for calculator in calculators:
            cleanup = getattr(calculator, "cleanup_wavecar", None)
            if cleanup is not None:
                cleanup()


def relax(atoms, calculator, directory="go", fmax=0.02, steps=500,
          resume=False, optimizer=FIRE, maxstep=0.1, keep_wavecar=False):
    """CP geometry optimization; delete WAVECAR after successful final output."""
    atoms, directory, loaded = _start(atoms, directory, resume)
    offset = int(atoms.info.get("step", 0)) if loaded else 0
    atoms.calc = calculator
    atoms.info["cp_stage"] = "geometry_optimization"
    opt = optimizer(atoms, logfile=str(directory / "optimization.log"), maxstep=maxstep)
    last_saved = [offset if loaded else -1]

    def record():
        step = offset + opt.nsteps
        force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        frame = snapshot(atoms, step=step, fmax_ev_per_a=force,
                         numerical_converged=force < fmax)
        if step != last_saved[0]:
            write_frames(directory / "trajectory.extxyz", frame, append=True)
            last_saved[0] = step
        write_frames(directory / "latest.extxyz", frame, atomic=True)

    try:
        opt.attach(record)
        remaining = max(0, steps - offset)
        if remaining:
            opt.run(fmax=fmax, steps=remaining)
        record()
        if np.linalg.norm(atoms.get_forces(), axis=1).max() >= fmax:
            raise RuntimeError("Geometry not converged within the step budget; checkpoint retained")
        write_frames(directory / "final.extxyz", snapshot(atoms, step=offset + opt.nsteps,
                     numerical_converged=True), atomic=True)
    finally:
        opt.close()
    _cleanup_wavecars([calculator], keep_wavecar)
    return atoms


def path_coordinate(images, mask=None):
    """Normalized arc length over selected atoms, with MIC image displacements."""
    if mask is None:
        mask = ~images[0].arrays["fixed_mask"]
    mask = np.asarray(mask)
    lengths = [0.0]
    for before, after in zip(images[:-1], images[1:]):
        delta = find_mic((after.positions - before.positions)[mask], before.cell, before.pbc)[0]
        lengths.append(lengths[-1] + float(np.linalg.norm(delta)))
    lengths = np.array(lengths)
    if lengths[-1] <= 0:
        raise ValueError("Zero-length path")
    return lengths / lengths[-1]


class _ParallelBand(NEB):
    """Prefetch independent SCFs with propagated exceptions, then use ASE NEB."""
    def __init__(self, images, workers, **kwargs):
        super().__init__(images, parallel=False, **kwargs)
        self.workers = workers

    def get_forces(self):
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(lambda image: image.get_forces(), self.images))
        return super().get_forces()


def cineb(initial, final, calculator_factory, directory="neb", n_images=6,
          band=None, relax_endpoints=True, endpoint_fmax=0.02, endpoint_steps=500,
          pre_fmax=0.05, fmax=0.03, pre_steps=150, steps=350, spring=0.1,
          workers=1, resume=False, reaction_mask=None, keep_wavecar=False):
    """Ordinary CP-NEB then CP-CI-NEB, with extxyz band checkpoints.

    factory(work_directory, image_index) returns one independent calculator.
    band is an optional complete initial band (including endpoints), e.g. the
    previous potential's path. All energies/forces are recalculated at this U.
    Endpoints are always checked against endpoint_fmax, including on resume.
    WAVECARs, including endpoints, are kept until the full CI-NEB succeeds.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if n_images < 1 or workers < 1:
        raise ValueError("n_images and workers must be positive")
    latest = root / "latest_band.extxyz"
    if latest.exists() and not resume:
        raise FileExistsError("Use resume=True or choose a new NEB directory")
    initial, final = initial.copy(), final.copy()
    check_identity(final, initial)
    endpoints = []
    for index, atoms in ((0, initial), (n_images + 1, final)):
        atoms.info.update(image_index=index, endpoint="IS" if index == 0 else "FS")
        work = root / ("image_%02d" % index)
        calc = calculator_factory(work / "vasp", index)
        if relax_endpoints:
            atoms = relax(atoms, calc, directory=work, fmax=endpoint_fmax,
                          steps=endpoint_steps, resume=resume, keep_wavecar=True)
        else:
            atoms.calc = calc
            if np.linalg.norm(atoms.get_forces(), axis=1).max() >= endpoint_fmax:
                raise RuntimeError("Endpoint is not stationary at the target potential")
        endpoints.append(atoms)
    resumed = latest.exists()
    if resumed:
        images = read_restart(latest, index=":")
    elif band is not None:
        images = [image.copy() for image in band]
    else:
        images = [endpoints[0].copy() for _ in range(n_images + 1)] + [endpoints[1].copy()]
    if len(images) != n_images + 2:
        raise ValueError("Band must contain n_images + 2 frames")
    for image in images:
        check_identity(image, initial)
    images[0], images[-1] = endpoints
    if not resumed and band is None:
        NEB(images, method="improvedtangent").interpolate(method="idpp", mic=True, apply_constraint=True)
    for index, image in enumerate(images[1:-1], 1):
        image.info["image_index"] = index
        image.calc = calculator_factory(root / ("image_%02d/vasp" % index), index)
    targets = [getattr(image.calc, "mu_e", None) for image in images]
    if len(set(targets)) != 1:
        raise ValueError("Every NEB image must use the same electron reservoir")
    completed_phase = images[1].info.get("neb_phase") if resumed else None
    for phase, climb, threshold, budget in (("neb", False, pre_fmax, pre_steps),
                                             ("ci_neb", True, fmax, steps)):
        if phase == "neb" and completed_phase == "ci_neb":
            continue
        offset = int(images[1].info.get("phase_step", 0)) if completed_phase == phase else 0
        for image in images:
            image.info["cp_stage"] = phase
        neb = _ParallelBand(images, workers=workers, climb=climb, k=spring, method="improvedtangent")
        opt = FIRE(neb, logfile=str(root / (phase + ".log")), dt=0.05, maxstep=0.1)
        last_saved = [-1]

        def record():
            forces = neb.get_forces().reshape(n_images, len(initial), 3)
            residual = float(np.linalg.norm(forces, axis=2).max())
            energies = [image.get_potential_energy() for image in images]
            coordinate = path_coordinate(images, reaction_mask)
            frames = []
            for index, image in enumerate(images):
                frame = snapshot(image, neb_phase=phase, phase_step=offset + opt.nsteps,
                                 image_index=index, n_images_total=len(images),
                                 reaction_coordinate=float(coordinate[index]),
                                 relative_grand_potential_ev=energies[index] - energies[0],
                                 neb_residual_ev_per_a=residual,
                                 numerical_converged=residual < threshold,
                                 label_eligibility="pending_branch_image_resolution_and_size_qc")
                frame.set_array("neb_forces", np.zeros((len(image), 3)) if index in (0, len(images)-1)
                                else forces[index - 1].copy())
                frames.append(frame)
            if opt.nsteps != last_saved[0]:
                write_frames(root / "bands.extxyz", frames, append=True)
                last_saved[0] = opt.nsteps
            write_frames(latest, frames, atomic=True)
            return frames, residual

        try:
            opt.attach(record)
            if budget > offset:
                opt.run(fmax=threshold, steps=budget - offset)
            frames, residual = record()
        finally:
            opt.close()
        if residual >= threshold:
            raise RuntimeError(phase + " not converged within its step budget; band checkpoint retained")
    energies = np.array([image.get_potential_energy() for image in images])
    for frame in frames:
        frame.info.update(barrier_ev=float(energies.max() - energies[0]),
                          reaction_energy_ev=float(energies[-1] - energies[0]),
                          highest_image=int(energies.argmax()),
                          resolved_interior_maximum=0 < energies.argmax() < len(images) - 1,
                          numerical_converged=True)
    write_frames(root / "final.extxyz", frames, atomic=True)
    _cleanup_wavecars([image.calc for image in images], keep_wavecar)
    return images


def _rng_state(rng):
    return dumps(rng.get_state())


def _run_md(atoms, calculator, directory, steps, timestep_fs, temperature_K,
            friction_per_fs, seed, resume, steering=None, keep_wavecar=False):
    atoms, root, loaded = _start(atoms, directory, resume)
    offset = int(atoms.info.get("step", 0)) if loaded else 0
    if steps < offset or timestep_fs <= 0 or friction_per_fs < 0:
        raise ValueError("Invalid MD budget, timestep or friction")
    spec = dict(timestep_fs=timestep_fs, temperature_K=temperature_K,
                friction_per_fs=friction_per_fs, seed=seed, steering=steering)
    # CV is serialized by its public bond definition, not its Python object.
    if steering is not None:
        spec["steering"] = dict(steering, cv=steering["cv"].terms, steps=steps)
    signature = dumps(spec)
    if loaded and atoms.info.get("dynamics_protocol_json", signature) != signature:
        raise ValueError("MD/steering protocol changed; use a new directory")
    rng = np.random.RandomState(seed)
    if loaded and "rng_state_json" in atoms.info:
        state = json.loads(atoms.info["rng_state_json"])
        rng.set_state((state[0], np.asarray(state[1], np.uint32), *state[2:]))
    elif temperature_K is not None and not atoms.has("momenta"):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    work = float(atoms.info.get("protocol_work_ev", 0)) if loaded else 0.0
    if steering is None:
        atoms.calc = calculator
    else:
        center = steering["start"] + (steering["stop"] - steering["start"]) * offset / steps
        atoms.calc = HarmonicBias(calculator, steering["cv"], center, steering["spring"])
    atoms.info["cp_stage"] = "slow_growth" if steering else "molecular_dynamics"
    if temperature_K is None:
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=str(root / "md.log"))
    else:
        dyn = Langevin(atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K,
                       friction=friction_per_fs / units.fs, fixcm=False, rng=rng,
                       logfile=str(root / "md.log"))

    def record(step, append=True):
        atoms.get_forces()  # ensure current-geometry SCF, but snapshot uses raw forces
        frame = snapshot(atoms, step=step, time_fs=step * timestep_fs,
                         kinetic_energy_ev=atoms.get_kinetic_energy(), protocol_work_ev=work,
                         dynamics_protocol_json=signature)
        frame.info.pop("rng_state_json", None)
        if append:
            write_frames(root / "trajectory.extxyz", frame, append=True)
        frame.info["rng_state_json"] = _rng_state(rng)
        write_frames(root / "latest.extxyz", frame, atomic=True)
        return frame

    try:
        frame = record(offset, append=not loaded or not (root / "trajectory.extxyz").exists())
        for step in range(offset + 1, steps + 1):
            if steering is not None:
                bias = atoms.calc
                center = steering["start"] + (steering["stop"] - steering["start"]) * step / steps
                value = steering["cv"].value_gradient(atoms)[0]
                work += 0.5 * bias.spring * ((value - center) ** 2 - (value - bias.center) ** 2)
                bias.set_center(center)
            dyn.run(1)
            frame = record(step)
        write_frames(root / "final.extxyz", frame, atomic=True)
    finally:
        dyn.close()
    _cleanup_wavecars([calculator], keep_wavecar)
    return atoms


def md(atoms, calculator, directory="md", steps=1000, timestep_fs=0.5,
       temperature_K=300.0, friction_per_fs=0.01, seed=0, resume=False, keep_wavecar=False):
    """CP Born-Oppenheimer MD: Langevin, or NVE when temperature_K=None."""
    return _run_md(atoms, calculator, directory, steps, timestep_fs, temperature_K,
                   friction_per_fs, seed, resume, keep_wavecar=keep_wavecar)


def slow_growth(atoms, calculator, cv, start, stop, directory="slow_growth",
                steps=1000, spring=10.0, timestep_fs=0.5, temperature_K=300.0,
                friction_per_fs=0.01, seed=0, resume=False, keep_wavecar=False):
    """CP steered MD with V=0.5*k*(CV-lambda)^2 and a linear lambda schedule.

    Store exact discrete protocol work at each fixed-geometry center update.
    A single finite-speed work trace is not an equilibrium free-energy profile.
    """
    if steps < 1 or spring <= 0:
        raise ValueError("Slow growth needs steps > 0 and spring > 0")
    return _run_md(atoms, calculator, directory, steps, timestep_fs, temperature_K,
                   friction_per_fs, seed, resume,
                   dict(cv=cv, start=float(start), stop=float(stop), spring=float(spring)),
                   keep_wavecar=keep_wavecar)
