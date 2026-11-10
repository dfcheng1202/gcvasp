"""Refine initial.extxyz at a new U using the installed gcvasp package.

Copy this file into each job directory and edit U_SHE_V below.
The input must contain one complete band, including both endpoints.
"""
import hashlib
import os
from pathlib import Path

from gcvasp import CPVasp, cineb, dimer_from_neb, read_restart
from gcvasp.io import dumps

# Edit this value in each of the four job directories --------------------
U_SHE_V = +0.30                 # +0.30, 0.00, -0.30, or -0.60 V vs SHE
MU_SHE_EV = -4.57               # user-selected reference for this test
POTENTIAL_TOL_V = 0.01

HERE = Path(__file__).resolve().parent
INPUT = HERE / "initial.extxyz"
OUTPUT = HERE / "neb"
RESUME = True
REFERENCE_ID = "test_mu_SHE_minus4p57_eV"

VASP_BINARY = "/path/to/vasp_std"
MPI_RANKS = int(os.environ.get("SLURM_NTASKS", "32"))
VASP_COMMAND = os.environ.get("VASP_COMMAND", f"mpirun -np {MPI_RANKS} {VASP_BINARY}")

# Same low-accuracy DFT settings as the original -0.4 V test.
DFT = dict(
    xc="PBE", pp="PBE", setups="minimal", gga="RP", ivdw=11,
    encut=350, kpts=(2, 2, 1), gamma=False,
    prec="Normal", lreal="Auto", lasph=True,
    ispin=1, isym=0, ismear=0, sigma=0.1,
    algo="All", ediff=1e-5, nelm=200, ncore=4,
    ldipol=False,
    lsol=True, lsol_scf=True, isol=2, eb_k=78.4, soltemp=298.0,
    c_molar=1.0, r_ion=4.0, lnldiel=True, lnlion=True,
    lwave=True, lcharg=False, laechg=False, lvtot=False, lvhar=False,
)
WORKERS = 1                    # one MPI VASP at a time
ENDPOINT_FMAX = 0.05           # eV/Angstrom
PRE_NEB_FMAX = 0.10
CI_NEB_FMAX = 0.05
ENDPOINT_STEPS = 400
PRE_NEB_STEPS = 200
CI_NEB_STEPS = 400
ENDPOINT_MODE = "joint"        # no separate IS/FS relaxation
RUN_DIMER = False              # optionally refine the internal maximum afterward
# -----------------------------------------------------------------------


def load_band(mu_e):
    band = read_restart(INPUT, index=":")
    if len(band) < 3:
        raise ValueError("initial.extxyz must contain a complete band with at least 3 images")
    source_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    for index, atoms in enumerate(band):
        parent = dict(atoms.info)
        atoms.calc = None
        # Retain UIDs, masks, roles and restored constraints; discard old labels.
        for name in ("forces", "neb_forces", "optimization_forces", "dimer_forces",
                     "dimer_mode", "bias_forces", "propagation_forces", "momenta"):
            atoms.arrays.pop(name, None)
        atoms.info = dict(
            parent_frame_info_json=dumps(parent),
            parent_nelect=parent.get("nelect"),
            input_file=str(INPUT), input_sha256=source_hash, input_image_index=index,
            potential_target_v=U_SHE_V, mu_target_ev=mu_e, mu_she_ev=MU_SHE_EV,
            potential_reference_id=REFERENCE_ID,
            she_reference_status="user_selected_test_value",
            scientific_status="low_accuracy_potential_transfer_test",
        )
    return band


def main():
    mu_e = MU_SHE_EV - U_SHE_V
    band = load_band(mu_e)

    def calculator(directory, image_index):
        settings = dict(DFT)
        guess = band[image_index].info.get("parent_nelect")
        if guess is not None:
            settings["nelect"] = float(guess)
        return CPVasp(
            mu_e=mu_e, mu_she=MU_SHE_EV, potential_reference_id=REFERENCE_ID,
            directory=directory, command=VASP_COMMAND,
            tolerance=POTENTIAL_TOL_V, warm_start=True, **settings,
        )

    print(f"U = {U_SHE_V:+.2f} V vs SHE; target mu_e = {mu_e:.2f} eV", flush=True)
    print(f"Input: {INPUT}; images: {len(band)}; output: {OUTPUT}", flush=True)
    # Joint mode moves endpoints with physical forces and interior images with NEB forces.
    # CPVasp uses EFERMI_REF=0 internally and adjusts NELECT to mu_e above.
    cineb(
        band[0], band[-1], calculator, directory=OUTPUT,
        n_images=len(band) - 2, band=band, endpoint_mode=ENDPOINT_MODE,
        endpoint_fmax=ENDPOINT_FMAX, endpoint_steps=ENDPOINT_STEPS,
        pre_fmax=PRE_NEB_FMAX, fmax=CI_NEB_FMAX,
        pre_steps=PRE_NEB_STEPS, steps=CI_NEB_STEPS,
        workers=WORKERS, resume=RESUME,
    )
    print("CP-CI-NEB complete:", OUTPUT / "final.extxyz", flush=True)
    if RUN_DIMER:
        dimer_from_neb(OUTPUT / "final.extxyz", calculator, directory=HERE / "dimer",
                       fmax=0.03, steps=200, resume=RESUME)


if __name__ == "__main__":
    main()
