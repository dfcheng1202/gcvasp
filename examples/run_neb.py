from gcvasp import read_restart as read
from gcvasp import cineb
from settings import calculator, RESUME

INITIAL = "initial_guess.extxyz"
FINAL = "final_guess.extxyz"
OUTPUT = "cp_neb"
PREVIOUS_BAND = None  # e.g. "previous_U/final.extxyz"; use a new OUTPUT for a new U
INTERNAL_IMAGES = 6
WORKERS = 1          # choose MPI ranks per VASP_COMMAND accordingly

if __name__ == "__main__":
    band = read(PREVIOUS_BAND, index=":") if PREVIOUS_BAND else None
    cineb(read(INITIAL), read(FINAL), calculator, directory=OUTPUT,
          n_images=INTERNAL_IMAGES, band=band, workers=WORKERS,
          endpoint_fmax=0.02, pre_fmax=0.05, fmax=0.03,
          endpoint_steps=500, pre_steps=150, steps=350, resume=RESUME)
