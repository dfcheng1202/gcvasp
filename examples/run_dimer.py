"""Refine an existing CI-NEB candidate with constant-potential dimer forces.

Copy this file and settings.py into the job directory. Set the same target
potential as the source band in settings.py, then run python run_dimer.py.
"""
from gcvasp import dimer_from_neb
from settings import calculator, RESUME

BAND = "neb/final.extxyz"       # latest_band.extxyz also accepts a rough band
OUTPUT = "cp_dimer"            # keep separate from the NEB directories
IMAGE_INDEX = None             # auto peak; explicitly choose if an endpoint is highest
FMAX = 0.03
STEPS = 200
DIMER_SEPARATION = 0.01        # Angstrom, center-to-probe distance


if __name__ == "__main__":
    dimer_from_neb(BAND, calculator, directory=OUTPUT, image_index=IMAGE_INDEX,
                   fmax=FMAX, steps=STEPS, dimer_separation=DIMER_SEPARATION,
                   resume=RESUME)
