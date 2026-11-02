"""Moving harmonic restraint; keep bias and physical grand-potential labels apart."""
import copy

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic


class BondCV:
    """A sum of bond lengths: [(i, j, coefficient), ...], indices start at zero.

    E.g. [(O, H, 1), (M, H, -1)] describes proton transfer.
    """
    def __init__(self, terms):
        self.terms = [(int(i), int(j), float(c)) for i, j, c in terms]
        if not self.terms:
            raise ValueError("Provide at least one bond")

    def value_gradient(self, atoms):
        value, gradient = 0.0, np.zeros((len(atoms), 3))
        for i, j, coefficient in self.terms:
            vector, distance = find_mic(atoms.positions[j] - atoms.positions[i], atoms.cell, atoms.pbc)
            if distance < 1e-10:
                raise ValueError("Zero-length CV bond")
            value += coefficient * float(distance)
            derivative = coefficient * vector / distance
            gradient[i] -= derivative
            gradient[j] += derivative
        return value, gradient


class HarmonicBias(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calculator, cv, center, spring):
        super().__init__()
        self.base, self.cv = calculator, cv
        self.center, self.spring = float(center), float(spring)
        self.cp_info = {}
        self.unbiased_results = {}

    def set_center(self, center):
        self.center = float(center)
        self.reset()  # base calculator retains the expensive SCF at this geometry

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        raw_forces = self.base.get_forces(atoms)
        omega = self.base.get_potential_energy(atoms, force_consistent=True)
        value, gradient = self.cv.value_gradient(atoms)
        delta = value - self.center
        bias = 0.5 * self.spring * delta ** 2
        self.bias_forces = -self.spring * delta * gradient
        self.unbiased_results = dict(energy=omega, free_energy=omega, forces=raw_forces.copy())
        self.results = dict(energy=omega + bias, free_energy=omega + bias,
                            forces=raw_forces + self.bias_forces)
        self.cp_info = copy.deepcopy(getattr(self.base, "cp_info", {}))
        self.cp_info.update(cv_value_a=value, restraint_center_a=self.center,
                            restraint_spring_ev_per_a2=self.spring,
                            bias_energy_ev=bias, propagation_energy_ev=omega + bias,
                            d_bias_d_center_ev_per_a=-self.spring * delta)
