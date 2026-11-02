from gcvasp import read_restart as read
from gcvasp import BondCV, slow_growth
from settings import calculator, RESUME

INPUT = "cp_go/final.extxyz"
OUTPUT = "cp_slow_growth"
CV_STOP_A = 1.0  # O-H minus metal-H; edit the endpoint and pulling speed
DONOR_O_INDEX = 65       # edit all three zero-based indices for your structure
TRANSFER_H_INDEX = 70
TARGET_METAL_INDEX = 58

if __name__ == "__main__":
    atoms = read(INPUT)
    cv = BondCV([(DONOR_O_INDEX, TRANSFER_H_INDEX, 1.0),
                 (TARGET_METAL_INDEX, TRANSFER_H_INDEX, -1.0)])
    slow_growth(atoms, calculator(OUTPUT + "/vasp"), cv,
                start=cv.value_gradient(atoms)[0], stop=CV_STOP_A, directory=OUTPUT,
                steps=2000, spring=10., timestep_fs=0.5, temperature_K=300,
                friction_per_fs=0.01, seed=42, resume=RESUME)
