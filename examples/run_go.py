from gcvasp import read_restart as read
from gcvasp import relax
from settings import calculator, RESUME

INPUT = "initial_guess.extxyz"
OUTPUT = "cp_go"

if __name__ == "__main__":
    atoms = read(INPUT)
    relax(atoms, calculator(OUTPUT + "/vasp"), directory=OUTPUT,
          fmax=0.02, steps=500, resume=RESUME)
