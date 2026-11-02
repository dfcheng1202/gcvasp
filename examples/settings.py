"""Edit this file before running; all paths are relative to the work folder."""
from gcvasp import CPVasp

MU_E_EV = -4.60     # example absolute mu_e; edit for the chosen interface/protocol
MU_SHE_EV = None   # insert your calibrated SHE Fermi energy if available
REFERENCE_ID = None
# With a calibrated reference: MU_E_EV = MU_SHE_EV - U_SHE_V
RESUME = False


def calculator(directory, image_index=0):
    return CPVasp(
        mu_e=MU_E_EV, mu_she=MU_SHE_EV, potential_reference_id=REFERENCE_ID,
        directory=directory, tolerance=0.005, warm_start=True,
        pp="PBE", gga="RP", ivdw=11, encut=400, ispin=1,
        kpts=(3, 3, 1), gamma=False, ismear=0, sigma=0.05,
        ediff=1e-6, nelm=200, c_molar=1.0, r_ion=4.0,
        lwave=True, lcharg=False, lvtot=False, lvhar=False,
    )
