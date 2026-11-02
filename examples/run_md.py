from gcvasp import read_restart as read
from gcvasp import md
from settings import calculator, RESUME

INPUT = "cp_go/final.extxyz"
OUTPUT = "cp_md"

if __name__ == "__main__":
    md(read(INPUT), calculator(OUTPUT + "/vasp"), directory=OUTPUT,
       steps=1000, timestep_fs=0.5, temperature_K=300,
       friction_per_fs=0.01, seed=42, resume=RESUME)
