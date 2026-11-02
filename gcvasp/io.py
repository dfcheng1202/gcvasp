"""extxyz records. Standard energy/forces always mean unbiased physical E/F."""
import copy
import json
import os
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, dict2constraint
from ase.io import read
from ase.io.extxyz import output_column_format


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, default=json_default, sort_keys=True)


def prepare_atoms(atoms):
    """Retain existing UIDs; supply index identities only when absent."""
    if "atom_uid" not in atoms.arrays:
        atoms.new_array("atom_uid", np.array(["atom:%06d" % i for i in range(len(atoms))]))
    mask = np.zeros(len(atoms), bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[constraint.get_indices()] = True
    if "fixed_mask" in atoms.arrays and not atoms.constraints:
        mask = atoms.arrays["fixed_mask"].astype(bool)
        atoms.set_constraint(FixAtoms(mask=mask))
    atoms.set_array("fixed_mask", mask)


def snapshot(atoms, calculator=None, **info):
    """Never call Atoms.get_forces(): FixAtoms/NEB/bias must not change labels."""
    calc = atoms.calc if calculator is None else calculator
    if calc is None:
        raise ValueError("A snapshot needs calculator results")
    physical = getattr(calc, "unbiased_results", calc.results)
    frame = atoms.copy()
    frame.calc = None
    frame.info = copy.deepcopy(atoms.info)
    for name in ("energy", "free_energy", "forces", "stress"):
        frame.info.pop(name, None)
        frame.arrays.pop(name, None)
    frame.info.update(copy.deepcopy(getattr(calc, "cp_info", {})))
    frame.info.update(info)
    frame.info["constraints_json"] = dumps([c.todict() for c in atoms.constraints])
    frame.info["force_kind"] = getattr(calc, "cp_info", {}).get("force_kind", "raw_physical_calculator")
    frame.info["energy_kind"] = "grand_potential_smearing_free_energy"
    frame.calc = SinglePointCalculator(frame, energy=float(physical["free_energy"]),
                                      free_energy=float(physical["free_energy"]),
                                      forces=np.asarray(physical["forces"]).copy())
    if hasattr(calc, "bias_forces"):
        frame.set_array("bias_forces", calc.bias_forces.copy())
        frame.set_array("propagation_forces", np.asarray(calc.results["forces"]).copy())
    return frame


def write_frames(path, frames, append=False, atomic=False):
    """extxyz with 16-decimal real columns for force labels and MD restarts.

    ASE's default 8-decimal columns lose velocity/position precision on resume.
    Use ASE's property/header formatter without globally modifying its writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + ".tmp") if atomic else path
    if hasattr(frames, "positions"):
        frames = [frames]
    with target.open("a" if append else "w") as stream:
        for source in frames:
            frame = source.copy()
            arrays = dict(frame.arrays, symbols=np.array(frame.get_chemical_symbols()))
            if source.calc is not None:
                for key in ("energy", "free_energy"):
                    if key in source.calc.results:
                        frame.info[key] = float(source.calc.results[key])
                if "forces" in source.calc.results:
                    arrays["forces"] = np.asarray(source.calc.results["forces"])
            if "fixed_mask" in arrays:
                arrays["move_mask"] = ~arrays["fixed_mask"].astype(bool)
            columns = ["symbols", "positions"] + [key for key in arrays
                         if key not in ("numbers", "symbols", "positions")]
            header, _, _, fmt = output_column_format(frame, columns, arrays)
            fmt = fmt.replace("%16.8f", "%24.16e")
            stream.write(str(len(frame)) + "\n" + header + "\n")
            for i in range(len(frame)):
                values = [value for key in columns for value in np.asarray(arrays[key][i]).reshape(-1)]
                stream.write(fmt % tuple(values))
    if atomic:
        os.replace(target, path)


def read_restart(path, index=-1):
    frames = read(str(path), index=index, format="extxyz")
    for atoms in frames if isinstance(frames, list) else [frames]:
        if "constraints_json" in atoms.info:
            atoms.set_constraint([dict2constraint(c) for c in json.loads(atoms.info["constraints_json"])])
    return frames


def check_identity(atoms, reference):
    """The small set of invariants required to resume or interpolate a band."""
    prepare_atoms(atoms)
    prepare_atoms(reference)
    if (not np.array_equal(atoms.numbers, reference.numbers)
            or not np.array_equal(atoms.arrays["atom_uid"], reference.arrays["atom_uid"])
            or not np.array_equal(atoms.arrays["fixed_mask"], reference.arrays["fixed_mask"])
            or not np.array_equal(atoms.pbc, reference.pbc)
            or not np.allclose(atoms.cell, reference.cell, atol=1e-8, rtol=0)):
        raise ValueError("Atom order/UIDs, fixed mask, cell or PBC differ")
    fixed = reference.arrays["fixed_mask"]
    if not np.allclose(atoms.positions[fixed], reference.positions[fixed], atol=1e-8, rtol=0):
        raise ValueError("Fixed atoms moved")
