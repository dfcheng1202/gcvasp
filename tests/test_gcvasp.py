"""Local software checks only: no VASP executable, PAWs, UMA or HPC."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HAS_ASE = importlib.util.find_spec("ase") is not None
PACKAGE = Path(__file__).resolve().parents[1]
if HAS_ASE:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.vasp import Vasp
    from ase.constraints import FixAtoms
    from ase.io import read
    with mock.patch.object(sys, "path", [str(PACKAGE)] + sys.path):
        from gcvasp import (BondCV, CPVasp, cineb, md, path_coordinate, read_electronic_state,
                            read_restart, relax, slow_growth, snapshot, write_frames)
        from gcvasp.bias import HarmonicBias

    def output(mu=-4.6, fermi=-4.6, free=-20.0, nelect=11.25, q=.25, converged=True):
        return """vasp.6.3.2 synthetic format fixture, not a scientific calculation
Nonlinear solvation model:
ISOL = 2
C_MOLAR = 1.0
EFERMI_ref = {mu}
 NELECT = 11.000 total number of electrons
 FERMI_SHIFT = 1000.0
 Adjusting number of electrons for constant potential calculation
 EFERMI = -4.70000000      EFERMI_ref = {mu}
 q_sol  = 0.20000000      -q*mu      = 0.92000000
 NELECT = 11.20000000     new NELECT = 11.25000000
 capacitance = 0.5
 Adjusting number of electrons for constant potential calculation
 EFERMI = {fermi}      EFERMI_ref = {mu}
 q_sol  = {q}         -q*mu      = 1.15000000
 NELECT = {nelect}    new NELECT = 11.24999999
 capacitance = 0.75
 E-fermi : {fermi} XC(G=0): 0.00
 aborting loop {reason}
 free  energy   TOTEN  = {free} eV
 energy  without entropy= {zero}  energy(sigma->0) = {zero}
""".format(mu=mu, fermi=fermi, free=free, zero=free+.01, nelect=nelect, q=q,
           reason="because EDIFF is reached" if converged else "because EDIFF was not reached (unconverged)")

    def fixed_output(nelect=11., fermi=-4.6, free=-21.15, converged=True):
        return """synthetic fixed-Ne VASPsol++ output; no native electron update
 NELECT = {nelect:.6f}
 Solvation model:
 LSOL = T
 LSOL_SCF = T
 ISOL = 2
 EFERMI_ref = 0.000000
 C_MOLAR = 1.0
 NELECT = {nelect:.4f} total number of electrons
 E-fermi : {fermi:.4f}
 aborting loop {reason}
 free energy TOTEN = {free:.12f} eV
 energy without entropy= {zero:.12f} energy(sigma->0) = {zero:.12f}
""".format(nelect=nelect, fermi=fermi, free=free, zero=free+.01,
           reason="because EDIFF is reached" if converged else "because EDIFF was not reached (unconverged)")

    def fixed_backend(root, capacitance=2.5, converged=True):
        """Analytic canonical F(R,N); dF/dN matches its printed Fermi level."""
        trials = []
        def calculate(self, atoms, properties, system_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            root.mkdir(parents=True, exist_ok=True)
            nelect = round(self.float_params.get("nelect") or 11., 6)
            q = nelect - 11.
            x = atoms.positions[-1, 0] - 4.
            mu0 = -4.7687 + .2*x
            free = -20. + .5*x*x + mu0*q + q*q/(2*capacitance)
            fermi = mu0 + q/capacitance
            forces = np.zeros((2, 3))
            forces[0, 0], forces[-1, 0] = 2., -x-.2*q
            trials.append(dict(nelect=nelect, positions=atoms.positions.copy(),
                               istart=self.int_params["istart"], icharg=self.int_params["icharg"]))
            (root / "OUTCAR").write_text(fixed_output(nelect, fermi, free, converged))
            (root / "POTCAR").write_text("synthetic PAW hash fixture; no licensed data")
            (root / "WAVECAR").write_text("synthetic wavefunction fixture")
            self.results = dict(energy=free+.01, free_energy=free, forces=forces)
            self.converged, self.version = converged, "synthetic_backend"
            self._store_param_state()
        return trials, calculate

    class ToyCP(Calculator):
        """Analytic double well with CP metadata for testing ionic algorithms."""
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, mu=-4.6):
            super().__init__()
            self.mu_e = mu
            self.cp_info = {}
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            x, y, z = atoms.positions[-1] - [5, 5, 8]
            omega = (x*x-1)**2 + .5*(y*y+z*z)
            forces = np.zeros((len(atoms), 3))
            forces[0] = [2, 0, 0]  # raw force remains nonzero on the fixed metal
            forces[-1] = [-4*x*(x*x-1), -y, -z]
            excess = .3*x + .2*self.mu_e
            self.results = dict(energy=omega, free_energy=omega, forces=forces)
            self.cp_info = dict(mu_target_ev=self.mu_e, fermi_ev=self.mu_e,
                                nelect=11+excess, neutral_nelect=11., charge_e=-excess,
                                grand_potential_ev=omega, fixed_ne_free_energy_ev=omega+excess*self.mu_e,
                                scf_converged=True, potential_converged=True,
                                force_kind="analytic_physical_force",
                                evidence="analytic_software_fixture_not_DFT")

    class WaveToy(ToyCP):
        """Exercise real WAVECAR deletion with synthetic, disposable files."""
        cleanup_wavecar = CPVasp.cleanup_wavecar

        def __init__(self, directory, observer=None):
            super().__init__()
            self.directory, self.observer = Path(directory), observer

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            directory = Path(self.directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "WAVECAR").write_text("synthetic wavefunction cache")
            if self.observer is not None:
                self.observer()

    def structure(x=4.):
        atoms = Atoms("AuH", positions=[[5, 5, 5], [x, 5, 8]], cell=[12, 12, 25], pbc=True)
        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.new_array("atom_uid", np.array(["fixture:Au", "fixture:H"]))
        return atoms


@unittest.skipUnless(HAS_ASE, "ASE is optional for the planning CLI")
class GCVaspTests(unittest.TestCase):
    def test_native_energy_charge_sign_and_final_count_are_not_initial_or_next_guess(self):
        result = read_electronic_state(output(), 11., mu_she=-4.6)
        self.assertEqual(result["nelect"], 11.25)
        self.assertEqual(result["q_sol_e"], .25)
        self.assertEqual(result["charge_e"], -.25)
        self.assertEqual(result["fermi_ev"], -4.6)  # FERMI_SHIFT is deliberately irrelevant
        self.assertEqual(result["potential_actual_v"], 0)
        self.assertEqual(result["grand_potential_ev"], -20)
        self.assertAlmostEqual(result["fixed_ne_free_energy_ev"], -21.15)
        self.assertAlmostEqual(result["grand_potential_zero_ev"], -19.99)
        self.assertEqual(result["charge_updates"], 2)
        trace = json.loads(result["electronic_trace_json"])
        self.assertEqual(trace[-1]["next_nelect"], 11.24999999)
        self.assertEqual(trace[-1]["capacitance_e_per_v"], .75)

    def test_fixed_ne_energy_conversion_is_once_at_target_reservoir(self):
        text = """EFERMI_ref = 0
 NELECT = 11.25 total number of electrons
 E-fermi : -4.60
 free energy TOTEN = -21.15 eV
 energy without entropy= -21.14 energy(sigma->0) = -21.14
 aborting loop because EDIFF is reached
"""
        result = read_electronic_state(text, 11., mu_target=-4.6)
        self.assertFalse(result["native_constant_potential"])
        self.assertAlmostEqual(result["grand_potential_ev"], -20)
        self.assertAlmostEqual(result["fixed_ne_free_energy_ev"], -21.15)
        self.assertIsNone(result["potential_actual_v"])
        self.assertFalse(read_electronic_state(output(converged=False), 11.)["scf_converged"])

    def test_external_cp_converges_at_fixed_geometry_and_transforms_canonical_energy_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calc = CPVasp(-4.2, directory=root, mu_she=-4.6, gga="RP", pp="PBE")
            atoms = structure()
            trials, fake_vasp = fixed_backend(root)
            atoms.calc = calc
            with mock.patch.object(Vasp, "calculate", fake_vasp), \
                    mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.):
                omega = atoms.get_potential_energy()
                q = calc.get_number_of_electrons() - 11.
                self.assertAlmostEqual(omega, -20.-.5*2.5*.5687**2, places=8)
                # E-fermi in this fixture has the real OUTCAR's four decimals.
                self.assertAlmostEqual(q, 2.5*.5687, delta=5e-4)
                self.assertEqual(np.linalg.norm(atoms.get_forces()[0]), 0)
                self.assertEqual(len(trials), 3)
                for trial in trials:
                    np.testing.assert_array_equal(trial["positions"], atoms.positions)
                self.assertEqual(trials[0]["istart"], 0)
                self.assertEqual(trials[1]["istart"], 1)
                frames = read(str(root / "evaluations.extxyz"), index=":")
                self.assertEqual([bool(a.info["potential_converged"]) for a in frames], [False, False, True])
                self.assertTrue(all(a.info["scf_converged"] for a in frames))
                self.assertAlmostEqual(frames[-1].info["fixed_ne_free_energy_ev"], omega + q*(-4.2), places=8)
                self.assertAlmostEqual(frames[-1].info["grand_potential_zero_ev"], omega+.01, places=8)
                self.assertEqual(frames[-1].get_forces(apply_constraint=False)[0, 0], 2.)
                self.assertEqual(frames[-1].info["electron_control"], "external_fixed_geometry")
                self.assertEqual(frames[-1].info["gcvasp_version"], "0.3.0")
                self.assertEqual(frames[-1].info["cpvasp_version"], "0.3.0")
                self.assertEqual(frames[-1].info["charge_iterations"], 3)
                self.assertEqual(frames[-1].info["charge_updates"], 2)
                self.assertEqual(len(json.loads(frames[-1].info["electronic_trace_json"])), 3)
                atoms.positions[-1, 0] += .1
                atoms.get_forces()
            self.assertEqual(len(trials), 5)  # previous Ne + learned capacitance: two trials
            self.assertAlmostEqual(trials[3]["nelect"], 11.+q, places=6)
            self.assertAlmostEqual(calc.float_params["nelect"], 11.+q-.05, delta=5e-4)
            self.assertEqual(calc.int_params["istart"], 1)
            self.assertEqual(calc.int_params["icharg"], 0)
            self.assertEqual(calc.float_params["efermi_ref"], 0.)
            self.assertEqual(calc.int_params["nsw"], 0)
            self.assertEqual(calc.int_params["ibrion"], -1)
            self.assertEqual(frames[0].info["nelect"], 11.)
            self.assertEqual(frames[0].arrays["atom_uid"].tolist(), ["fixture:Au", "fixture:H"])
            restored = CPVasp(-4.2, directory=root, mu_she=-4.6, gga="RP", pp="PBE")
            self.assertAlmostEqual(restored._capacitance, 2.5, delta=1e-3)
            with self.assertRaisesRegex(ValueError, "different protocol"):
                CPVasp(-4.7, directory=root, mu_she=-4.6, gga="RP", pp="PBE")

    def test_external_cp_force_is_the_gradient_of_omega_after_electron_equilibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atoms = structure()
            atoms.calc = CPVasp(-4.2, directory=root)
            _, fake_vasp = fixed_backend(root)
            with mock.patch.object(Vasp, "calculate", fake_vasp), \
                    mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.):
                force = atoms.get_forces()[-1, 0]
                atoms.positions[-1, 0] += .05
                plus = atoms.get_potential_energy()
                atoms.positions[-1, 0] -= .1
                minus = atoms.get_potential_energy()
            self.assertAlmostEqual(force, -(plus-minus)/.1, delta=1e-4)

    def test_charge_search_bounds_each_electron_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atoms = structure()
            atoms.calc = CPVasp(-4.2, directory=root, capacitance_init=10., max_electron_step=.25)
            trials, fake_vasp = fixed_backend(root)
            with mock.patch.object(Vasp, "calculate", fake_vasp), \
                    mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.):
                atoms.get_forces()
            self.assertTrue(atoms.calc.cp_info["potential_converged"])
            self.assertLessEqual(np.abs(np.diff([trial["nelect"] for trial in trials])).max(), .25+1e-9)

    def test_go_with_external_cp_keeps_only_potential_converged_ionic_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calc = CPVasp(-4.2, directory=root/"vasp")
            _, fake_vasp = fixed_backend(root/"vasp")
            with mock.patch.object(Vasp, "calculate", fake_vasp), \
                    mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.), \
                    redirect_stdout(io.StringIO()):
                relax(structure(), calc, directory=root/"go", fmax=.01, steps=100)
            frames = read(str(root/"go/trajectory.extxyz"), index=":")
            self.assertGreater(len(frames), 1)
            self.assertFalse((root/"vasp/WAVECAR").exists())
            self.assertTrue((root/"vasp/evaluations.extxyz").exists())
            for frame in frames:
                self.assertTrue(frame.info["potential_converged"] and frame.info["scf_converged"])
                self.assertEqual(frame.info["ensemble"], "electron_grand_canonical")
                self.assertEqual(frame.info["efermi_ref_ev"], 0.)
                self.assertEqual(frame.get_forces(apply_constraint=False)[0, 0], 2.)

    def test_fixed_ne_parser_retains_echo_precision_and_never_accepts_unupdated_native_run(self):
        result = read_electronic_state(fixed_output(nelect=11.123456), 11., mu_target=-4.2)
        self.assertEqual(result["nelect"], 11.123456)
        self.assertEqual(result["q_sol_source"], "fixed_nelect_minus_neutral")
        # This mirrors the reported NSW=0 failure: a negative reference was read,
        # but no native charge update ran. It must not become a valid CP label.
        missed = fixed_output(nelect=729., fermi=-4.7687).replace("EFERMI_ref = 0.000000", "EFERMI_ref = -4.200000")
        with self.assertRaisesRegex(ValueError, "No native"):
            read_electronic_state(missed, 729., mu_target=-4.2)

    def test_nonconverged_potential_is_recorded_but_never_returned_to_optimizer(self):
        for electronic in (True, False):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                calc = CPVasp(-4.2, directory=root, max_charge_steps=2)
                atoms = structure()
                atoms.calc = calc
                trials, fake_vasp = fixed_backend(root, capacitance=float("inf"), converged=electronic)
                with mock.patch.object(Vasp, "calculate", fake_vasp), \
                        mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.):
                    with self.assertRaisesRegex(RuntimeError, "convergence failed"):
                        atoms.get_forces()
                self.assertFalse(calc.results)
                self.assertEqual(len(trials), 2 if electronic else 1)
                frame = read_restart(root / "evaluations.extxyz")
                self.assertFalse(frame.info["potential_converged"] and frame.info["scf_converged"])

    def test_vaspsol_parameter_registration_and_static_input(self):
        with tempfile.TemporaryDirectory() as temp:
            calc = CPVasp(-4.6, directory=temp, capacitance_init=1., soltemp=298.)
            atoms = structure()
            calc.spinpol = False  # minimal unpolarized fixture; no PAW lookup
            calc.sort = np.arange(len(atoms))
            calc.write_incar(atoms, directory=temp)
            text = (Path(temp) / "INCAR").read_text()
            self.assertIn("EFERMI_REF = 0.000000", text)
            self.assertIn("NSW = 0", text)
            self.assertIn("IBRION = -1", text)
            self.assertEqual(calc._capacitance, 1.)
            with self.assertRaisesRegex(ValueError, "mu_e"):
                CPVasp(0)
            with self.assertRaisesRegex(ValueError, "ISOL"):
                CPVasp(-4.6, isol=1)

    def test_failed_vasp_output_is_kept_across_retries_without_fake_energy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atoms = structure()
            atoms.calc = CPVasp(-4.6, directory=root)
            for attempt in range(2):
                (root / "OUTCAR").write_text("failed SCF attempt %d" % attempt)
                with mock.patch.object(Vasp, "calculate", side_effect=RuntimeError("VASP process failed")):
                    with self.assertRaisesRegex(RuntimeError, "VASP process failed"):
                        atoms.get_forces()
            failures = sorted((root / "failures").iterdir())
            self.assertEqual(len(failures), 2)
            for attempt, directory in enumerate(failures):
                self.assertIn("attempt %d" % attempt, (directory / "OUTCAR").read_text())
                frame = read_restart(directory / "geometry.extxyz")
                self.assertIsNone(frame.calc)
                self.assertEqual(frame.info["label_eligibility"], "excluded_failed_evaluation")

    def test_go_checkpoint_restart_and_raw_forces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial = structure(3.8)
            result = relax(initial, ToyCP(), directory=root, steps=150)
            self.assertLess(np.linalg.norm(result.get_forces(), axis=1).max(), .02)
            frame = read_restart(root / "final.extxyz")
            self.assertEqual(frame.get_forces(apply_constraint=False)[0, 0], 2)
            self.assertEqual(frame.info["force_kind"], "analytic_physical_force")
            before = len(read(str(root / "trajectory.extxyz"), index=":"))
            relax(initial, ToyCP(), directory=root, steps=150, resume=True)
            self.assertEqual(len(read(str(root / "trajectory.extxyz"), index=":")), before)
            with self.assertRaises(FileExistsError):
                relax(initial, ToyCP(), directory=root)

    def test_go_wavecar_retention_is_explicit_and_failed_go_keeps_its_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kept = WaveToy(root/"kept/vasp")
            relax(structure(4.), kept, directory=root/"kept", keep_wavecar=True)
            self.assertTrue((root/"kept/vasp/WAVECAR").exists())
            failed = WaveToy(root/"failed/vasp")
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                relax(structure(3.8), failed, directory=root/"failed", steps=0)
            self.assertTrue((root/"failed/vasp/WAVECAR").exists())
            self.assertTrue((root/"failed/latest.extxyz").exists())
            self.assertFalse((root/"failed/final.extxyz").exists())

    def test_neb_keeps_endpoint_wavecars_until_the_entire_band_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for keep in (False, True):
                work = root/str(keep)
                observations = []
                def observe():
                    observations.append(all((work/('image_%02d/vasp/WAVECAR' % i)).exists() for i in (0, 5)))
                def factory(directory, index):
                    return WaveToy(directory, observe if index not in (0, 5) else None)
                cineb(structure(4.), structure(6.), factory, directory=work, n_images=4,
                      pre_steps=120, steps=250, keep_wavecar=keep)
                self.assertTrue(observations and all(observations))
                self.assertTrue((work/"final.extxyz").exists())
                for index in range(6):
                    self.assertEqual((work/('image_%02d/vasp/WAVECAR' % index)).exists(), keep)
            failed = root/"failed"
            with self.assertRaisesRegex(RuntimeError, "ci_neb not converged"):
                cineb(structure(4.), structure(6.), lambda directory, index: WaveToy(directory),
                      directory=failed, n_images=4, steps=0)
            self.assertTrue((failed/"latest_band.extxyz").exists())
            self.assertFalse((failed/"final.extxyz").exists())
            for index in range(6):
                self.assertTrue((failed/('image_%02d/vasp/WAVECAR' % index)).exists())

    def test_md_and_steering_clean_wavecar_only_after_reaching_the_time_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unrelated = root/"WAVECAR"
            unrelated.write_text("another calculation's cache")
            cv = BondCV([(0, 1, 1)])
            start = cv.value_gradient(structure(3.8))[0]
            for steering in (False, True):
                for keep in (False, True):
                    work = root/('%s_%s' % (steering, keep))
                    calc = WaveToy(work/"vasp")
                    extra = dict(cv=cv, start=start, stop=start+.02) if steering else {}
                    function = slow_growth if steering else md
                    function(structure(3.8), calc, directory=work, steps=2, temperature_K=None,
                             keep_wavecar=keep, **extra)
                    self.assertEqual((work/"vasp/WAVECAR").exists(), keep)
                    frame = read_restart(work/"final.extxyz")
                    self.assertEqual(frame.info["step"], 2)
                    self.assertTrue((work/"latest.extxyz").exists())
            self.assertEqual(unrelated.read_text(), "another calculation's cache")

    def test_interrupted_md_and_steering_preserve_wavecar_and_last_complete_step(self):
        class Interrupted(WaveToy):
            def calculate(self, *args, **kwargs):
                if self.calls:
                    raise RuntimeError("synthetic interruption before the next MD step")
                return super().calculate(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cv = BondCV([(0, 1, 1)])
            start = cv.value_gradient(structure(3.8))[0]
            for steering in (False, True):
                work = root/str(steering)
                extra = dict(cv=cv, start=start, stop=start+.02) if steering else {}
                function = slow_growth if steering else md
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    function(structure(3.8), Interrupted(work/"vasp"), directory=work,
                             steps=2, temperature_K=None, **extra)
                self.assertTrue((work/"vasp/WAVECAR").exists())
                self.assertEqual(read_restart(work/"latest.extxyz").info["step"], 0)
                self.assertFalse((work/"final.extxyz").exists())

    def test_cp_cineb_uses_physical_omega_and_preserves_a_complete_extxyz_band(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = cineb(structure(4.), structure(6.), lambda directory, index: ToyCP(),
                           directory=root, n_images=4, workers=2, pre_steps=120, steps=250)
            frames = read_restart(root / "final.extxyz", index=":")
            self.assertEqual(len(frames), 6)
            self.assertAlmostEqual(frames[0].info["barrier_ev"], 1., delta=.01)
            self.assertLess(frames[0].info["neb_residual_ev_per_a"], .03)
            self.assertTrue(frames[0].info["resolved_interior_maximum"])
            for frame in frames:
                self.assertEqual(frame.get_forces(apply_constraint=False)[0, 0], 2)
                self.assertEqual(frame.arrays["atom_uid"].tolist(), ["fixture:Au", "fixture:H"])
                self.assertIn("neb_forces", frame.arrays)
                self.assertEqual(frame.info["mu_target_ev"], -4.6)
            coordinate = path_coordinate(images)
            self.assertTrue((np.diff(coordinate) > 0).all())
            self.assertEqual((coordinate[0], coordinate[-1]), (0, 1))
            self.assertAlmostEqual(coordinate[frames[0].info["highest_image"]], .5, delta=.01)
            cineb(structure(4.), structure(6.), lambda directory, index: ToyCP(), directory=root,
                  n_images=4, workers=2, pre_steps=120, steps=250, resume=True)

    def test_cineb_rejects_nonstationary_endpoint_and_mismatched_potential(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "Endpoint"):
                cineb(structure(3.7), structure(6.), lambda directory, index: ToyCP(),
                      directory=Path(temp)/"bad_endpoint", relax_endpoints=False)
            with self.assertRaisesRegex(ValueError, "reservoir"):
                cineb(structure(4.), structure(6.), lambda directory, index: ToyCP(-4.6-.1*index),
                      directory=Path(temp)/"bad_potential", n_images=2)

    def test_ci_budget_failure_keeps_checkpoint_and_resume_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            factory = lambda directory, index: ToyCP()
            with self.assertRaisesRegex(RuntimeError, "ci_neb not converged"):
                cineb(structure(4.), structure(6.), factory, directory=root, n_images=4, steps=0)
            self.assertTrue((root / "latest_band.extxyz").exists())
            self.assertFalse((root / "final.extxyz").exists())
            cineb(structure(4.), structure(6.), factory, directory=root, n_images=4, steps=250, resume=True)
            self.assertTrue((root / "final.extxyz").exists())

    def test_parallel_image_exception_reaches_caller(self):
        class BrokenCP(ToyCP):
            def calculate(self, *args, **kwargs):
                raise RuntimeError("synthetic SCF failure")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "synthetic SCF failure"):
                cineb(structure(4.), structure(6.), lambda directory, index: BrokenCP() if index == 2 else ToyCP(),
                      directory=temp, n_images=3, workers=2)
            self.assertFalse((Path(temp) / "final.extxyz").exists())

    def test_bond_cv_and_bias_force_match_energy_derivatives_and_keep_raw_labels(self):
        atoms = structure(4.)
        cv = BondCV([(0, 1, 1)])
        value, gradient = cv.value_gradient(atoms)
        epsilon = 1e-6
        shifted = atoms.copy()
        shifted.positions[1, 0] += epsilon
        self.assertAlmostEqual((cv.value_gradient(shifted)[0]-value)/epsilon, gradient[1, 0], places=5)
        calc = HarmonicBias(ToyCP(), cv, center=value+.2, spring=5.)
        atoms.calc = calc
        force = atoms.get_forces(apply_constraint=False)[1, 0]
        e1 = atoms.get_potential_energy()
        atoms.positions[1, 0] += epsilon
        e2 = atoms.get_potential_energy()
        self.assertAlmostEqual(-(e2-e1)/epsilon, force, places=4)
        frame = snapshot(atoms)
        self.assertAlmostEqual(frame.get_potential_energy(), calc.unbiased_results["free_energy"])
        np.testing.assert_allclose(frame.get_forces(apply_constraint=False), calc.unbiased_results["forces"])
        np.testing.assert_allclose(frame.arrays["propagation_forces"], calc.results["forces"])

    def test_md_extxyz_restart_preserves_velocities_rng_and_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            md(structure(), ToyCP(), directory=root/"resumed", steps=3, temperature_K=20, seed=42)
            resumed = md(structure(), ToyCP(), directory=root/"resumed", steps=6,
                         temperature_K=20, seed=42, resume=True)
            direct = md(structure(), ToyCP(), directory=root/"direct", steps=6, temperature_K=20, seed=42)
            np.testing.assert_allclose(resumed.positions, direct.positions, atol=1e-12)
            np.testing.assert_allclose(resumed.get_momenta(), direct.get_momenta(), atol=1e-12)
            frames = read(str(root/"resumed/trajectory.extxyz"), index=":")
            self.assertEqual([frame.info["step"] for frame in frames], list(range(7)))
            self.assertEqual(frames[-1].info["time_fs"], 3.)
            self.assertEqual(frames[-1].get_forces(apply_constraint=False)[0, 0], 2)

    def test_resume_flag_on_new_md_directory_does_not_import_parent_step(self):
        with tempfile.TemporaryDirectory() as temp:
            initial = structure()
            initial.info["step"] = 84
            md(initial, ToyCP(), directory=temp, steps=2, temperature_K=None, resume=True)
            frames = read(str(Path(temp) / "trajectory.extxyz"), index=":")
            self.assertEqual([frame.info["step"] for frame in frames], [0, 1, 2])

    def test_slow_growth_work_schedule_and_unbiased_force_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cv = BondCV([(0, 1, 1)])
            start = cv.value_gradient(structure())[0]
            slow_growth(structure(), ToyCP(), cv, start, start+.02, directory=root,
                        steps=4, spring=5., temperature_K=None)
            frames = read(str(root/"trajectory.extxyz"), index=":")
            expected_work = 0.
            for before, after in zip(frames[:-1], frames[1:]):
                old_center, new_center = before.info["restraint_center_a"], after.info["restraint_center_a"]
                value = before.info["cv_value_a"]
                expected_work += .5*5.*((value-new_center)**2-(value-old_center)**2)
                self.assertAlmostEqual(after.info["protocol_work_ev"], expected_work)
                self.assertEqual(after.get_forces(apply_constraint=False)[0, 0], 2)
                self.assertIn("bias_forces", after.arrays)
                self.assertAlmostEqual(after.get_potential_energy()+after.info["bias_energy_ev"],
                                       after.info["propagation_energy_ev"])
            self.assertAlmostEqual(frames[-1].info["restraint_center_a"], start+.02)
            self.assertEqual(len(frames), 5)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                slow_growth(structure(), ToyCP(), cv, start, start+.03, directory=root,
                            steps=4, spring=5., temperature_K=None, resume=True)


if __name__ == "__main__":
    unittest.main()
