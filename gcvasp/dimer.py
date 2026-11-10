"""ASE dimer searches on the same constant-potential surface as CI-NEB."""
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
try:
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
except ImportError:
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

from .io import check_identity, dumps, read_restart, snapshot, write_frames
from .workflows import _cleanup_wavecars, _start


def _unit_mode(atoms, mode):
    mode = np.asarray(mode, dtype=float).copy()
    if mode.shape != (len(atoms), 3) or not np.isfinite(mode).all():
        raise ValueError("Dimer mode must be a finite (natoms, 3) array")
    # Project onto the input constraints without moving the starting structure.
    trial = atoms.copy()
    trial.set_positions(atoms.positions + 1e-6 * mode)
    mode = (trial.positions - atoms.positions) / 1e-6
    length = np.linalg.norm(mode)
    if length < 1e-12:
        raise ValueError("Dimer direction is zero after applying constraints")
    return mode / length


def dimer_guess(band, image_index=None):
    """Return a seed with a normalized, MIC-aware neighboring-band tangent.

    band is a complete single band, or an extxyz filename. An unconverged
    latest_band.extxyz is allowed. Auto selection requires an interior global
    maximum; if an endpoint is highest, select an interior image explicitly.
    The seed is a guess, not an accepted transition state.
    """
    source = str(Path(band).resolve()) if isinstance(band, (str, Path)) else None
    images = read_restart(band, index=":") if source else band
    if len(images) < 3:
        raise ValueError("A dimer seed needs a band with at least three images")
    for index, atoms in enumerate(images):
        check_identity(atoms, images[0])
        if (atoms.info.get("n_images_total", len(images)) != len(images)
                or atoms.info.get("image_index", index) != index):
            raise ValueError("Supply one complete band, not concatenated NEB history")
    targets = {getattr(image.calc, "mu_e", image.info.get("mu_target_ev")) for image in images}
    if len(targets) != 1:
        raise ValueError("The source band must use one electron reservoir")
    energies = np.array([image.get_potential_energy() for image in images])
    if not np.isfinite(energies).all():
        raise ValueError("Source band energies must be finite")
    highest = int(energies.argmax())
    index = highest if image_index is None else image_index
    if not isinstance(index, (int, np.integer)) or not 0 < index < len(images) - 1:
        raise ValueError("Highest energy is at an endpoint or index is invalid; select an interior image_index")
    seed = images[index].copy()
    seed.info.update(getattr(images[index].calc, "cp_info", {}))
    forward = find_mic(images[index + 1].positions - seed.positions, seed.cell, seed.pbc)[0]
    backward = find_mic(seed.positions - images[index - 1].positions, seed.cell, seed.pbc)[0]
    # Improved tangent: high-energy side receives the larger energy weight.
    left, center, right = energies[index - 1:index + 2]
    if right > center > left:
        tangent = forward
    elif right < center < left:
        tangent = backward
    else:
        large, small = sorted((abs(right - center), abs(left - center)), reverse=True)
        tangent = (large * forward + small * backward if right > left
                   else small * forward + large * backward)
        if large == 0:
            tangent = forward + backward
    for name in ("neb_forces", "optimization_forces", "bias_forces", "propagation_forces"):
        seed.arrays.pop(name, None)
    seed.set_array("dimer_mode", _unit_mode(seed, tangent))
    parent_info = dumps(seed.info)
    for key in ("neb_phase", "phase_step", "image_index", "n_images_total",
                "reaction_coordinate", "relative_grand_potential_ev", "neb_residual_ev_per_a",
                "interior_residual_ev_per_a", "endpoint_residual_ev_per_a", "endpoint_mode",
                "barrier_ev", "reaction_energy_ev", "highest_image", "resolved_interior_maximum"):
        seed.info.pop(key, None)
    seed.info.update(parent_frame_info_json=parent_info,
                     dimer_seed_image=index, source_band=source,
                     source_band_highest_image=highest,
                     source_band_numerical_converged=images[index].info.get("numerical_converged", False),
                     mu_target_ev=next(iter(targets)), cp_stage="dimer_seed",
                     numerical_converged=False,
                     label_eligibility="unoptimized_dimer_seed")
    return seed


class _DimerControl(DimerControl):
    def __init__(self, **kwargs):
        # ASE keeps defaults in a mutable class dictionary; isolate this run.
        self.parameters = DimerControl.parameters.copy()
        super().__init__(**kwargs)


class _CenterMinModeAtoms(MinModeAtoms):
    """Retain center labels before ASE temporarily evaluates displaced atoms."""
    def calculate_real_forces_and_energies(self, **kwargs):
        self.atoms.info["dimer_evaluation"] = "center"
        super().calculate_real_forces_and_energies(**kwargs)
        self.center_frame = snapshot(self.atoms)

    def get_potential_energy(self, force_consistent=False):
        # ASE 3.22's Optimizer passes force_consistent to MinModeAtoms.
        return super().get_potential_energy()

    def get_forces(self, real=False, pos=None, **kwargs):
        if real and pos is not None:
            old = self.atoms.get_positions()
            self.atoms.info["dimer_evaluation"] = "displaced"
            try:
                return super().get_forces(real=real, pos=pos, **kwargs)
            finally:
                self.atoms.set_positions(old)
                self.atoms.info["dimer_evaluation"] = "center"
        return super().get_forces(real=real, pos=pos, **kwargs)


class _CPDimerTranslate(MinModeTranslate):
    """Require physical as well as projected per-atom forces to converge."""
    def _physical_converged(self):
        return np.linalg.norm(self.atoms.get_forces(real=True), axis=1).max() < self.fmax

    def converged(self, *args, **kwargs):
        return super().converged(*args, **kwargs) and self._physical_converged()

    def gradient_converged(self, gradient):
        # ASE 3.29 calls this hook directly; ASE 3.22 uses converged().
        return super().gradient_converged(gradient) and self._physical_converged()


def dimer(atoms, calculator, directory="dimer", mode=None, fmax=0.03, steps=200,
          dimer_separation=0.01, maxstep=0.1, f_rot_min=0.01, f_rot_max=0.1,
          max_num_rot=3, use_central_forces=True, resume=False, keep_wavecar=False):
    """Refine a seed with CP forces at the center and every dimer probe.

    Pass mode explicitly or use atoms.arrays['dimer_mode'] from dimer_guess.
    Resume restores positions and direction, restarting the translation state.
    Negative curvature and small forces establish numerical convergence only;
    saddle order and chemical connectivity still require separate validation.
    Returns the final Atoms with cached physical center results.
    """
    atoms, root, loaded = _start(atoms, directory, resume)
    offset = int(atoms.info.get("step", 0)) if loaded else 0
    if fmax <= 0 or steps < offset or dimer_separation <= 0 or maxstep <= 0 or max_num_rot < 1:
        raise ValueError("Invalid dimer tolerance, displacement, rotation count or step budget")
    target = getattr(calculator, "mu_e", None)
    source_target = atoms.info.get("mu_target_ev")
    if source_target is not None and target is not None and not np.isclose(source_target, target, atol=1e-10, rtol=0):
        raise ValueError("Dimer seed and calculator use different electron reservoirs")
    parameters = dict(dimer_separation=dimer_separation, maximum_translation=maxstep,
                      f_rot_min=f_rot_min, f_rot_max=f_rot_max, max_num_rot=max_num_rot,
                      use_central_forces=use_central_forces, extrapolate_forces=False,
                      order=1, eigenmode_method="dimer", trial_angle=np.pi / 4,
                      trial_trans_step=0.001, cg_translation=True)
    signature = dumps(dict(mu_target_ev=target, **parameters))
    if loaded and atoms.info.get("dimer_protocol_json") != signature:
        raise ValueError("Dimer protocol changed; use a new directory")
    if loaded or mode is None:
        mode = atoms.arrays.get("dimer_mode")
    if mode is None:
        raise ValueError("Supply a dimer mode or construct a seed with dimer_guess")
    mode = _unit_mode(atoms, mode)
    atoms.set_array("dimer_mode", mode)
    atoms.info.update(cp_stage="dimer", numerical_converged=False)
    # An evaluated NEB image provides a useful first electronic guess.
    if isinstance(getattr(calculator, "cp_info", None), dict) and not calculator.cp_info:
        nelect = atoms.info.get("nelect")
        if nelect is not None and getattr(calculator, "_previous", None) is None:
            calculator.set(nelect=float(nelect))
    atoms.calc = calculator
    if not loaded:
        seed = atoms.copy()
        seed.info.update(dimer_protocol_json=signature, step=0)
        write_frames(root / "initial.extxyz", seed, atomic=True)
    last_saved = [offset if loaded else -1]
    with _DimerControl(logfile=str(root / "rotation.log"), eigenmode_logfile=None,
                       **parameters) as control:
        search = _CenterMinModeAtoms(atoms, control=control, eigenmodes=[mode])
        with _CPDimerTranslate(search, logfile=str(root / "dimer.log")) as opt:
            def record():
                projected = search.get_forces()
                frame = search.center_frame
                force = float(np.linalg.norm(search.get_forces(real=True), axis=1).max())
                residual = float(np.linalg.norm(projected, axis=1).max())
                curvature = float(search.get_curvature())
                if not np.isfinite([force, residual, curvature]).all():
                    raise RuntimeError("Nonfinite dimer forces or curvature; checkpoint retained")
                step = offset + opt.nsteps
                frame.set_array("dimer_mode", search.get_eigenmode().copy())
                frame.set_array("dimer_forces", projected.copy())
                frame.info.update(step=step, cp_stage="dimer", dimer_protocol_json=signature,
                                  fmax_ev_per_a=force, dimer_residual_ev_per_a=residual,
                                  dimer_curvature_ev_per_a2=curvature,
                                  numerical_converged=force < fmax and residual < fmax and curvature < 0,
                                  saddle_order_verified=False,
                                  label_eligibility="pending_saddle_order_and_connectivity_qc")
                if step != last_saved[0]:
                    write_frames(root / "trajectory.extxyz", frame, append=True)
                    last_saved[0] = step
                write_frames(root / "latest.extxyz", frame, atomic=True)
                return frame

            opt.attach(record)
            if steps > offset:
                opt.run(fmax=fmax, steps=steps - offset)
            frame = record()
            if not frame.info["numerical_converged"]:
                raise RuntimeError("Dimer not converged within the step budget; checkpoint retained")
            write_frames(root / "final.extxyz", frame, atomic=True)
    _cleanup_wavecars([calculator], keep_wavecar)
    return frame


def dimer_from_neb(band, calculator_factory, directory="dimer", image_index=None, **kwargs):
    """Build a band-tangent seed and refine it using a fresh calculator directory."""
    seed = dimer_guess(band, image_index=image_index)
    calc = calculator_factory(Path(directory) / "vasp", seed.info["dimer_seed_image"])
    return dimer(seed, calc, directory=directory, **kwargs)
