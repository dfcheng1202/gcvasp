"""Fixed-geometry electron-number control for ASE-driven VASPsol++ dynamics."""
import hashlib
from pathlib import Path
import re
import shutil
import time

import ase
import numpy as np
from ase.calculators.vasp import Vasp, create_input

from .io import dumps, prepare_atoms, read_restart, snapshot, write_frames

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def last_value(text, pattern):
    found = re.findall(pattern, text, flags=re.I | re.M)
    if not found:
        raise ValueError("Missing VASPsol++ output: " + pattern)
    return number(found[-1])


def read_electronic_state(text, neutral_nelect, mu_target=None, mu_she=None):
    """Parse one ASE-driven ionic single point, not a multi-ionic OUTCAR.

    q_sol is electron-positive. In the update block, NELECT is the evaluated
    count; 'new NELECT' is the next SCF guess and must never label this energy.
    The initial OUTCAR 'total number of electrons' can remain stale in CP mode.
    """
    applied = last_value(text, r"\bEFERMI_ref\s*=\s*(" + NUMBER + ")")
    target = applied if mu_target is None else float(mu_target)
    native = applied < 0
    fermi = last_value(text, r"E-fermi\s*:\s*(" + NUMBER + ")")
    free = last_value(text, r"free\s+energy\s+TOTEN\s*=\s*(" + NUMBER + ")")
    zero = last_value(text, r"energy\(sigma->0\)\s*=\s*(" + NUMBER + ")")
    trace = []
    for block in text.split("Adjusting number of electrons for constant potential calculation")[1:]:
        # Only this update block, before the next SCF/energy section.
        ef = last_value(block.split("capacitance")[0], r"\bEFERMI\s*=\s*(" + NUMBER + ")")
        q = last_value(block.split("capacitance")[0], r"\bq_sol\s*=\s*(" + NUMBER + ")")
        pair = re.search(r"^\s*NELECT\s*=\s*(" + NUMBER + r")\s+new NELECT\s*=\s*(" + NUMBER + ")", block, re.M | re.I)
        if pair is None:
            raise ValueError("Incomplete VASPsol++ electron-update block")
        cap = re.search(r"capacitance\s*=\s*(" + NUMBER + ")", block, re.I)
        trace.append(dict(fermi_ev=ef, q_sol_e=q, nelect=number(pair[1]),
                          next_nelect=number(pair[2]),
                          capacitance_e_per_v=number(cap[1]) if cap else None))
    if native:
        if not trace:
            raise ValueError("No native VASPsol++ electron-update output")
        nelect, q_sol = trace[-1]["nelect"], trace[-1]["q_sol_e"]
    else:
        # The echoed INCAR retains more digits than the four-decimal header.
        explicit = re.findall(r"^\s*NELECT\s*=\s*(" + NUMBER + r")[ \t]*$", text, re.M | re.I)
        nelect = number(explicit[-1]) if explicit else last_value(
            text, r"^\s*NELECT\s*=\s*(" + NUMBER + r")\s+total number of electrons")
        q_sol = nelect - neutral_nelect
    # VASPsol++ already prints F - q_sol*EFERMI_ref, including on E0.
    # Undo that term only to expose fixed-Ne E/F labels, then transform once
    # to the requested reservoir. Native calculations at their target are unchanged.
    helmholtz = free + q_sol * applied
    energy0 = zero + q_sol * applied
    omega = helmholtz - q_sol * target
    omega0 = energy0 - q_sol * target
    reasons = [line.lower() for line in text.splitlines() if "aborting loop" in line.lower()]
    scf = ("because ediff is reached" in reasons[-1]) if reasons else None
    return dict(mu_target_ev=target, efermi_ref_ev=applied, fermi_ev=fermi,
                potential_target_v=(mu_she - target) if mu_she is not None else None,
                potential_actual_v=(mu_she - fermi) if mu_she is not None else None,
                mu_she_ev=mu_she, fermi_error_ev=fermi - target,
                neutral_nelect=float(neutral_nelect), nelect=nelect,
                excess_electrons=nelect - neutral_nelect,
                charge_e=neutral_nelect - nelect, q_sol_e=q_sol,
                density_electron_count=neutral_nelect + q_sol if native else None,
                q_sol_source="density_update_block" if native else "fixed_nelect_minus_neutral",
                printed_free_energy_ev=free, printed_energy_zero_ev=zero,
                fixed_ne_free_energy_ev=helmholtz, fixed_ne_energy_zero_ev=energy0,
                grand_potential_ev=omega, grand_potential_zero_ev=omega0,
                energy_gauge="F_minus_excess_electrons_times_mu",
                native_constant_potential=native, scf_converged=scf,
                electronic_trace_json=dumps(trace),
                charge_updates=len(trace),
                potentiostat_capacitance_e_per_v=trace[-1]["capacitance_e_per_v"] if trace else None)


def register_parameters():
    """Register patched INCAR tags in this process, including ASE 3.22."""
    groups = dict(int_keys=("isol",),
                  float_keys=("efermi_ref", "efermi_tol", "capacitance_init", "c_molar",
                              "r_ion", "soltemp", "eb_k", "r_solv", "r_diel", "nsol_k"),
                  bool_keys=("lsol", "lsol_scf", "lnldiel", "lnlion"))
    for group, tags in groups.items():
        keys = getattr(create_input, group)
        for tag in tags:
            if tag not in keys:
                keys.append(tag)


class CPVasp(Vasp):
    """Converge the electron reservoir before returning force-consistent Omega.

    mu_e: negative vacuum-referenced electron chemical potential in eV.
    mu_she: optional *calibrated* SHE Fermi energy, solely for U reporting.
    warm_start reuses each image's last electron number and existing WAVECAR.
    Each VASP run is fixed-Ne, EFERMI_ref=0, NSW=0. A bounded secant search
    adjusts Ne at unchanged coordinates; ionic optimizers see converged CP only.
    All charge trials and physical forces go to directory/evaluations.extxyz.
    """
    def __init__(self, mu_e, directory="vasp", mu_she=None, potential_reference_id=None,
                 tolerance=0.005, warm_start=True, record_file=None,
                 max_charge_steps=16, capacitance_init=1.0, max_electron_step=1.0, **kwargs):
        register_parameters()
        if not np.isfinite(mu_e) or mu_e >= 0 or tolerance <= 0:
            raise ValueError("CP requires mu_e < 0 eV and tolerance > 0")
        if max_charge_steps < 1 or capacitance_init <= 0 or max_electron_step <= 0:
            raise ValueError("Charge-search budget, capacitance and electron step must be positive")
        if kwargs.get("charge") not in (None, 0) or kwargs.get("net_charge") not in (None, 0):
            raise ValueError("Use nelect as an initial guess; CPVasp controls the charge")
        kwargs.pop("charge", None)
        kwargs.pop("net_charge", None)
        settings = dict(lsol=True, lsol_scf=True, isol=2, c_molar=1.0, r_ion=4.0,
                        lnldiel=True, lnlion=True, ediff=1e-6, nelm=200,
                        lwave=warm_start, lcharg=False)
        settings.update(kwargs)
        if settings.get("nsw", 0) != 0 or settings.get("ibrion", -1) != -1:
            raise ValueError("ASE owns the ions: use IBRION=-1, NSW=0")
        if not settings["lsol"] or not settings["lsol_scf"] or settings["isol"] != 2 or settings["c_molar"] <= 0:
            raise ValueError("VASPsol++ CP requires LSOL, LSOL_SCF, ISOL=2, C_MOLAR>0")
        if settings.get("efermi_ref", 0) != 0:
            raise ValueError("Set mu_e for the target; external CP requires EFERMI_ref=0")
        settings.update(efermi_ref=0.0, ibrion=-1, nsw=0)
        self.mu_e, self.mu_she = float(mu_e), mu_she
        self.tolerance, self.warm_start = tolerance, warm_start
        self.max_charge_steps, self.max_electron_step = int(max_charge_steps), float(max_electron_step)
        self._capacitance = float(capacitance_init)
        self.record_file = Path(record_file) if record_file else Path(directory) / "evaluations.extxyz"
        self.cp_info = {}
        self._previous = None
        # The changing electron/WAVECAR starting guesses do not change the PES.
        # Retain the legacy electronic-protocol tag from cpvasp 0.2.0.
        # Renaming the package does not change the calculation or its checkpoints.
        self.protocol_json = dumps(dict(
            settings=settings, mu_target_ev=self.mu_e, mu_she=mu_she, tolerance=tolerance,
            potential_reference_id=potential_reference_id, electron_control="external_fixed_geometry",
            ase_version=ase.__version__, cpvasp_version="0.2.0"))
        self.protocol_hash = hashlib.sha256(self.protocol_json.encode()).hexdigest()
        self.potential_reference_id = potential_reference_id
        self._calls = 0
        if self.record_file.exists():
            last = read_restart(self.record_file)
            if last.info.get("cp_protocol_hash") != self.protocol_hash:
                raise ValueError("Existing calculations use a different protocol; choose a new directory")
            self._calls = int(last.info["force_call"]) + 1
            if last.info.get("scf_converged"):
                self._previous = dict(last.info)
                self._capacitance = last.info.get("potentiostat_capacitance_e_per_v") or self._capacitance
        super().__init__(directory=str(directory), **settings)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return super().get_potential_energy(atoms, force_consistent=True)

    def cleanup_wavecar(self):
        """Remove this calculator's wavefunction cache after task completion."""
        (Path(self.directory) / "WAVECAR").unlink(missing_ok=True)

    def read_convergence(self, lines=None):
        lines = self.load_file("OUTCAR") if lines is None else lines
        reasons = [line.lower() for line in lines if "aborting loop" in line.lower()]
        if reasons:
            return "because ediff is reached" in reasons[-1]
        return super().read_convergence(lines=lines)

    def get_number_of_electrons(self, lines=None):
        if lines is None and self.cp_info:
            return self.cp_info["nelect"]
        return super().get_number_of_electrons(lines=lines)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        try:
            return self._calculate_cp(atoms, properties, system_changes)
        except Exception as error:
            # Keep failed SCF evidence before the next restart overwrites OUTCAR.
            failure = Path(self.directory) / "failures" / str(time.time_ns())
            failure.mkdir(parents=True)
            for name in ("INCAR", "OUTCAR", "vasp.out", "vasprun.xml"):
                source = Path(self.directory) / name
                if source.exists():
                    shutil.copy2(source, failure / name)
            if atoms is not None:
                frame = atoms.copy()
                frame.calc = None
                frame.info.update(error=str(error), mu_target_ev=self.mu_e,
                                  label_eligibility="excluded_failed_evaluation")
                write_frames(failure / "geometry.extxyz", frame)
            self.results.clear()
            raise

    def _calculate_cp(self, atoms, properties, system_changes):
        from ase.calculators.calculator import all_changes
        if atoms is not None:
            prepare_atoms(atoms)
        if self.warm_start and self._previous is not None:
            self.set(nelect=self._previous["nelect"])
        trace, lower, upper = [], None, None
        for iteration in range(self.max_charge_steps):
            started = time.monotonic()
            if self.warm_start:
                wave = Path(self.directory) / "WAVECAR"
                restart_wave = wave.exists() and wave.stat().st_size > 0
                self.set(istart=1 if restart_wave else 0, icharg=0 if restart_wave else 2)
            super().calculate(atoms, properties, all_changes if system_changes is None else system_changes)
            info = self._record_charge_trial(trace, iteration, started)
            nelect, error = info["nelect"], info["fermi_error_ev"]
            if not info["scf_converged"]:
                raise RuntimeError("Electronic convergence failed; charge trial retained in evaluations.extxyz")
            if info["potential_converged"]:
                self._previous = info
                return
            if error < 0:
                lower = nelect
            else:
                upper = nelect
            guess = nelect - self._capacitance * error
            if lower is not None and upper is not None:
                left, right = sorted((lower, upper))
                if not left < guess < right:
                    guess = (left + right) / 2
            guess = nelect + np.clip(guess - nelect, -self.max_electron_step, self.max_electron_step)
            if guess <= 0:
                raise RuntimeError("Charge search requires NELECT > 0")
            if iteration + 1 < self.max_charge_steps:
                self.set(nelect=float(guess))
        raise RuntimeError("Potential convergence failed after %d charge trials; records retained" % self.max_charge_steps)

    def _record_charge_trial(self, trace, iteration, started):
        output = (Path(self.directory) / "OUTCAR").read_bytes()
        text = output.decode(errors="replace")
        info = read_electronic_state(text, self.default_nelect_from_ppp(), self.mu_e, self.mu_she)
        if info["efermi_ref_ev"] != 0:
            raise RuntimeError("Expected fixed-Ne VASPsol++ output with EFERMI_ref=0")
        if info["scf_converged"] is None:
            info["scf_converged"] = bool(self.converged)
        info["potential_converged"] = bool(abs(info["fermi_error_ev"]) <= self.tolerance)
        if trace:
            delta_mu = info["fermi_ev"] - trace[-1]["fermi_ev"]
            delta_n = info["nelect"] - trace[-1]["nelect"]
            if abs(delta_mu) > 1e-4 and delta_n / delta_mu > 0:
                self._capacitance = delta_n / delta_mu
        trace.append({key: info[key] for key in (
            "nelect", "fermi_ev", "fermi_error_ev", "printed_free_energy_ev",
            "grand_potential_ev", "scf_converged", "potential_converged")})
        info.update(force_call=self._calls, cp_protocol_hash=self.protocol_hash,
                    cp_protocol_json=self.protocol_json,
                    ensemble="electron_grand_canonical" if info["scf_converged"] and info["potential_converged"] else "fixed_electron_number",
                    backend_ensemble="fixed_electron_number",
                    electron_control="external_fixed_geometry", charge_iteration=iteration,
                    charge_iterations=len(trace), charge_updates=iteration,
                    electronic_trace_json=dumps(trace), potential_tolerance_ev=self.tolerance,
                    potentiostat_capacitance_e_per_v=self._capacitance,
                    potential_reference_id=self.potential_reference_id,
                    ase_version=ase.__version__, vasp_version=self.version,
                    gcvasp_version="0.4.0", cpvasp_version="0.3.0",
                    outcar_sha256=hashlib.sha256(output).hexdigest(),
                    source_directory=str(Path(self.directory).resolve()),
                    nelect_initial_guess=self.float_params.get("nelect"),
                    force_call_wall_seconds=time.monotonic() - started,
                    force_kind="raw_physical_dft", source_kind="vaspsolpp_dft",
                    potcar_sha256=hashlib.sha256((Path(self.directory) / "POTCAR").read_bytes()).hexdigest(),
                    label_eligibility="pending_protocol_branch_and_split_qc")
        self.cp_info = info
        # Tangents, image ranking, FIRE and MD must all use the same functional.
        self.results["energy"] = self.results["free_energy"] = info["grand_potential_ev"]
        if not np.isfinite(self.results["forces"]).all():
            raise RuntimeError("Nonfinite physical forces")
        write_frames(self.record_file, snapshot(self.atoms, calculator=self), append=True)
        self._calls += 1
        print("CP: Ne=%.6f E_F=%.5f target=%.5f dmu=%+.5f eV SCF=%s" % (
            info["nelect"], info["fermi_ev"], self.mu_e, info["fermi_error_ev"], info["scf_converged"]), flush=True)
        return info
