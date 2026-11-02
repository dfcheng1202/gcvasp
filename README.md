# gcvasp

Small ASE workflows for constant-potential calculations with VASPsol++:

- Geometry optimization.
- NEB followed by climbing-image NEB.
- Born-Oppenheimer MD with Langevin or NVE dynamics.
- Slow growth with a moving harmonic restraint.

All trajectories, electronic evaluations, and atomic checkpoints use extxyz.
The package was previously developed as `cpvasp`; the Python import is now
`gcvasp`. The calculator class remains `CPVasp`.

## Installation

Use an existing environment with Python >=3.8, ASE >=3.22.1, and NumPy >=1.21:

```bash
git clone https://github.com/dfcheng1202/gcvasp.git
cd gcvasp
python -m pip install --no-deps .
python -c "import gcvasp; print(gcvasp.__version__)"
```

`--no-deps` preserves the installed ASE and NumPy versions. VASPsol++, its
VASP executable, and PAW datasets must already be available on the calculation
host. They are not distributed in this repository.

Set the executable command and the parent directory of your PAW datasets:

```bash
export VASP_COMMAND="mpirun -np 32 /path/to/vasp_std"
export VASP_PP_PATH=/path/to/PAW_parent
```

Run Python once per job; the calculator starts the MPI VASP processes.

## Quick start

```python
from ase.io import read
from gcvasp import CPVasp, relax

atoms = read("initial.extxyz")
calc = CPVasp(
    mu_e=-4.20, directory="go/vasp",
    pp="PBE", gga="RP", ivdw=11,
    encut=400, kpts=(3, 3, 1), ispin=1,
    ismear=0, sigma=0.05, ediff=1e-6,
)
relax(atoms, calc, directory="go", fmax=0.02, steps=500, resume=True)
```

`mu_e` is the target electron chemical potential in eV relative to the solution
reference. With a compatible SHE reference, use
`mu_e = mu_she - U_SHE` in eV and volts. Supply `mu_she` and
`potential_reference_id` to retain that reference in the output.
The value -4.57 eV in the VASPsol++ documentation belongs to its stated
BEEF-vdW setup; calibrate the reference for the functional and interface used.

The editable examples cover GO, NEB, MD, and slow growth. Copy the relevant
script together with `examples/settings.py` into a calculation directory and
edit the input filenames and settings before running.

## Electron-number control

ASE updates the ions. At each new geometry, `CPVasp`:

1. Uses the previous electron count and available WAVECAR as starting guesses.
2. Runs a fixed-electron-number VASPsol++ SCF at unchanged atomic coordinates.
3. Checks `abs(E_F - mu_e) <= tolerance`.
4. If needed, updates `NELECT` using a bounded secant search and repeats the SCF.

The initial electron-response estimate is 1 electron/V. It is updated from
`delta_NELECT / delta_E_F` within the fixed-geometry search. Each adjustment
is capped at 1 electron, with at most 16 SCFs per search by default.
Bracketed searches use a midpoint if the secant estimate leaves the bracket.

Default potential tolerance is 0.005 eV. MD checks the potential at each new
geometry but changes the electron number only when the error exceeds the
tolerance. An SCF failure or an exhausted potential-search budget raises an
error before the optimizer or dynamics receives new forces.

Each VASP call uses `EFERMI_REF=0`, `IBRION=-1`, and `NSW=0`; Python controls
the target electron chemical potential. This avoids relying on native charge
updates being called in a static VASP run. Do not pass a negative
`efermi_ref` to `CPVasp`; set `mu_e` instead.

The backend requires VASPsol++ with `LSOL`, `LSOL_SCF`, `ISOL=2`, and nonzero
ionic screening through `C_MOLAR`. No legacy VASPsol `FERMI_SHIFT` is added.

## Energy and charge conventions

For the fixed-Ne backend, VASP prints the canonical smearing free energy F.
The package applies one reservoir transformation:

```text
q_sol = Ne - Ne_neutral          # excess electrons; electron-positive
charge_e = Ne_neutral - Ne       # net cell charge; positive means a deficit
Omega = F - q_sol * mu_e
```

Both ASE `energy` and `free_energy` return the unbiased, force-consistent
Omega derived from TOTEN. Zero-smearing energies are retained separately.
Do not add a second Legendre or CHE correction to these grand potentials.
The electron reference contributes a constant shift relative to a total-Ne
gauge; same-composition barriers at one potential are unaffected by that shift.

The parser also supports separate native VASPsol++ outputs containing complete
electron-update blocks and accounts for their existing reservoir correction.
`read_electronic_state` parses output; parsing alone does not equilibrate charge.

## NEB and potential continuation

```python
from gcvasp import CPVasp, cineb, read_restart

band = read_restart("initial.extxyz", index=":")

def factory(directory, image_index=0):
    return CPVasp(mu_e=-4.57, directory=directory,
                  pp="PBE", gga="RP", ivdw=11,
                  encut=400, kpts=(3, 3, 1))

cineb(band[0], band[-1], factory, directory="neb",
      n_images=len(band)-2, band=band, resume=True)
```

Both endpoints are first relaxed at the target potential and then held fixed.
All internal images are optimized as a coupled band, first with ordinary NEB
and then with CI-NEB. Providing `band` preserves its internal geometries;
without a band, the workflow generates IDPP images. Internal images are not
independently minimized into endpoint structures. ASE supplies the NEB method;
VTST is not required.

`examples/run_neb_from_band.py` is a self-contained continuation example.
Place it beside a complete `initial.extxyz` band, edit `U_SHE_V`, and set
`VASP_COMMAND`. It archives the source metadata, retains the source electron
counts as initial guesses, clears old result labels, and recomputes the band
at the new potential. Use a separate output directory for each potential.
Only one final band should be supplied, rather than concatenated NEB history.
Endpoint displacements are not automatically propagated into internal images.

`workers=1` evaluates images serially. For multiple workers, divide the allocated
MPI ranks among their VASP commands. Actual speedups depend on the system and
scheduler and must be measured on the calculation host.

## MD and slow growth

`md` defaults to Langevin dynamics; `temperature_K=None` selects NVE Velocity
Verlet. Timesteps are in fs and `friction_per_fs` is in inverse fs.

`slow_growth` moves the center of `0.5 * spring * (CV - center)**2` linearly.
`BondCV([(O, H, 1.0), (M, H, -1.0)])` defines an O-H minus metal-H distance
coordinate using zero-based atom indices. Physical and bias forces are recorded
separately, together with the center, observed CV, and discrete protocol work.
A single finite-speed pulling trajectory is not an equilibrium free-energy
profile.

The package respects input constraints; it does not choose frozen metal layers,
water anchors, proton identities, or reaction branches automatically.

## Output and restart

| File | Contents |
| --- | --- |
| `vasp/evaluations.extxyz` | Every parsed SCF trial, including intermediate charge searches |
| `trajectory.extxyz` | Ionic optimization or MD frames |
| `bands.extxyz` | Complete NEB bands grouped by phase, step, and image index |
| `latest.extxyz` / `latest_band.extxyz` | Atomically replaced restart checkpoints |
| `final.extxyz` | Final structure or complete converged band |
| `vasp/failures/` | Failed-run evidence and geometry for diagnosis |

Frames retain Fermi level, target and actual potential when a SHE reference is
supplied, electron count, neutral electron reference, net cell charge, canonical
and grand-canonical energies, physical forces, electronic-search history,
protocol and PAW hashes, software versions, atom UIDs, and constraints.
MD checkpoints additionally retain momenta and RNG state.

Standard `forces` are raw physical forces before fixed-atom, NEB, or bias
modifications. `neb_forces`, `bias_forces`, and `propagation_forces` are separate
arrays. Intermediate charge trials are marked fixed-Ne; electronically and
potential-converged frames are marked electron-grand-canonical.
The cell charge is not an atomic charge partitioning.

Use `resume=True` with the same input, potential, and protocol. Step budgets
are totals including completed work; NEB has separate ordinary and CI budgets.
GO and NEB restore positions and electronic guesses but restart FIRE state.
MD restores momenta and RNG state. Failed records are preserved.

The electronic protocol retains the legacy `cpvasp_version="0.2.0"` tag for
checkpoint compatibility. Actual release metadata is written as
`gcvasp_version` and the retained legacy `cpvasp_version` result field.
Matching cpvasp 0.2.x electronic records remain readable with the same ASE
version, settings, potential, and command. Existing scripts should change their
imports to `gcvasp` when switching distributions.

WAVECAR is overwritten as a working cache during a calculation. It is removed
only after successful final output for the complete GO, CI-NEB, MD, or
slow-growth task. Failures and interruptions retain it; NEB endpoint caches
remain until the entire band succeeds. Set `keep_wavecar=True` to retain it.
Direct calculator users can call `calculator.cleanup_wavecar()` at completion.

## Local software checks

```bash
python -m unittest discover -s tests -v
```

Tests use analytic potentials, synthetic OUTCAR text, and mock VASP calls.
They cover electron control, energy/force consistency, physical labels, NEB,
MD, slow growth, restart, failures, and WAVECAR lifetime. These checks do not
establish a physical barrier, finite-temperature activation free energy, or
HPC performance result. No real calculation data or licensed VASP files are
included.

## References

- [VASPsol++ documentation](https://github.com/VASPsol/VASPsol).
- [Native charge updates and static calculations](https://github.com/VASPsol/VASPsol/issues/6#issuecomment-2181870689).
- [VASP WAVECAR documentation](https://vasp.at/wiki/WAVECAR).
- [EChO](https://github.com/zishengz/echo/tree/9363a04c2cb52c3e0f4438a369a7846b09ac8cac), which inspired the workflow; this package was implemented independently.
