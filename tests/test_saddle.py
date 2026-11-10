"""Analytic tests of moving endpoints and constant-potential dimer plumbing."""
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_gcvasp import HAS_ASE

if HAS_ASE:
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.calculators.calculator import all_changes
    from ase.calculators.vasp import Vasp
    from gcvasp import (CPVasp, cineb, dimer, dimer_from_neb, dimer_guess,
                        read_restart, snapshot, write_frames)
    from gcvasp.workflows import _ParallelBand
    from gcvasp.dimer import DimerControl
    from test_gcvasp import ToyCP, WaveToy, structure, fixed_output

    def band_fixture():
        band = []
        for index, x in enumerate([4., 4.55, 5.12, 5.65, 6.]):
            atoms = structure(x)
            atoms.calc = ToyCP()
            atoms.get_forces()
            band.append(snapshot(atoms, image_index=index, n_images_total=5))
        return band


@unittest.skipUnless(HAS_ASE, "ASE required")
class SaddleTests(unittest.TestCase):
    def test_joint_endpoints_move_in_first_band_step_and_keep_raw_forces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial, final = structure(3.8), structure(6.25)
            band = [structure(x) for x in (3.8, 4.2, 4.7, 5.1, 5.7, 6.25)]
            images = cineb(initial, final, lambda directory, index: ToyCP(),
                           directory=root, n_images=4, endpoint_mode="joint", workers=2,
                           pre_steps=250, steps=300, band=band)
            history = read_restart(root / "bands.extxyz", index=":")
            first, second = history[:6], history[6:12]
            self.assertAlmostEqual(first[0].positions[1, 0], 3.8)
            self.assertAlmostEqual(first[-1].positions[1, 0], 6.25)
            self.assertNotEqual(second[0].positions[1, 0], first[0].positions[1, 0])
            self.assertNotEqual(second[-1].positions[1, 0], first[-1].positions[1, 0])
            self.assertNotEqual(second[1].positions[1, 0], first[1].positions[1, 0])
            self.assertFalse((root / "image_00/optimization.log").exists())
            np.testing.assert_allclose(images[0].positions[0], initial.positions[0])
            self.assertAlmostEqual(images[0].positions[1, 0], 4., delta=.01)
            self.assertAlmostEqual(images[-1].positions[1, 0], 6., delta=.01)
            frames = read_restart(root / "final.extxyz", index=":")
            for frame in frames:
                self.assertLess(frame.info["endpoint_residual_ev_per_a"], .03)
                self.assertLess(frame.info["neb_residual_ev_per_a"], .03)
                self.assertEqual(frame.get_forces(apply_constraint=False)[0, 0], 2.)
                np.testing.assert_allclose(frame.arrays["optimization_forces"][0], 0.)
            np.testing.assert_allclose(first[0].arrays["neb_forces"], 0.)
            self.assertGreater(np.linalg.norm(first[0].arrays["optimization_forces"]), .1)

    def test_joint_endpoint_forces_are_unprojected_and_enter_convergence(self):
        images = [structure(x) for x in (3.8, 4.5, 5.5, 6.2)]
        for atoms in images:
            atoms.calc = ToyCP()
        neb = _ParallelBand(images, workers=1, move_endpoints=True, method="improvedtangent")
        forces = neb.get_forces().reshape(4, 2, 3)
        np.testing.assert_allclose(forces[0], images[0].get_forces())
        np.testing.assert_allclose(forces[-1], images[-1].get_forces())
        self.assertEqual(len(neb), 8)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "neb not converged"):
                cineb(images[0], images[-1], lambda directory, index: ToyCP(), directory=temp,
                      n_images=2, band=images, endpoint_mode="joint", pre_steps=0)
            self.assertFalse((Path(temp) / "final.extxyz").exists())

    def test_joint_restart_retains_moved_endpoints_and_rejects_mode_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            factory = lambda directory, index: WaveToy(directory)
            with self.assertRaisesRegex(RuntimeError, "neb not converged"):
                cineb(structure(3.8), structure(6.2), factory, directory=root,
                      n_images=4, endpoint_mode="joint", pre_steps=1)
            before = read_restart(root / "latest_band.extxyz", index=":")
            self.assertTrue((root / "image_00/vasp/WAVECAR").exists())
            with self.assertRaisesRegex(ValueError, "mode changed"):
                cineb(structure(3.8), structure(6.2), factory, directory=root,
                      n_images=4, resume=True)
            with self.assertRaisesRegex(RuntimeError, "neb not converged"):
                cineb(structure(3.8), structure(6.2), factory, directory=root,
                      n_images=4, endpoint_mode="joint", pre_steps=1, resume=True)
            after = read_restart(root / "latest_band.extxyz", index=":")
            for a, b in zip(before, after):
                np.testing.assert_allclose(a.positions, b.positions)
            cineb(structure(3.8), structure(6.2), factory, directory=root,
                  n_images=4, endpoint_mode="joint", pre_steps=250, steps=300, resume=True)
            self.assertFalse((root / "image_00/vasp/WAVECAR").exists())

    def test_guess_selects_internal_peak_and_mic_tangent_with_fixed_mask(self):
        band = band_fixture()
        band[3].positions[1, 0] += 12
        band[3].calc = SinglePointCalculator(band[3], energy=.333, forces=np.zeros((2, 3)))
        seed = dimer_guess(band)
        self.assertEqual(seed.info["dimer_seed_image"], 2)
        self.assertFalse(seed.info["numerical_converged"])
        np.testing.assert_allclose(seed.arrays["dimer_mode"], [[0, 0, 0], [1, 0, 0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seed.extxyz"
            write_frames(path, seed)
            restored = read_restart(path)
            np.testing.assert_allclose(restored.arrays["dimer_mode"], seed.arrays["dimer_mode"])
            self.assertEqual(restored.constraints[0].get_indices().tolist(), [0])

    def test_guess_requires_explicit_interior_choice_if_endpoint_highest(self):
        band = band_fixture()
        band[-1].calc = SinglePointCalculator(band[-1], energy=20, forces=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "endpoint"):
            dimer_guess(band)
        seed = dimer_guess(band, image_index=2)
        self.assertEqual(seed.info["source_band_highest_image"], 4)
        with self.assertRaisesRegex(ValueError, "complete band"):
            dimer_guess(band + band)

    def test_live_band_seed_uses_current_calculator_labels_after_potential_transfer(self):
        band = [structure(x) for x in (4., 5.1, 6.)]
        for atoms in band:
            atoms.info.update(mu_target_ev=-4.6, barrier_ev=123., numerical_converged=True)
            atoms.calc = ToyCP(-4.2)
            atoms.get_forces()
        seed = dimer_guess(band)
        self.assertEqual(seed.info["mu_target_ev"], -4.2)
        self.assertEqual(seed.info["nelect"], band[1].calc.cp_info["nelect"])
        self.assertNotIn("barrier_ev", seed.info)
        self.assertFalse(seed.info["numerical_converged"])

    def test_dimer_reaches_saddle_and_records_center_labels_not_probe_labels(self):
        class ProbeCP(WaveToy):
            def calculate(self, atoms=None, *args, **kwargs):
                super().calculate(atoms, *args, **kwargs)
                self.cp_info["evaluated_x"] = float(atoms.positions[-1, 0])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = dimer_from_neb(band_fixture(), lambda directory, index: ProbeCP(directory),
                                    directory=root, steps=100)
            self.assertAlmostEqual(result.positions[1, 0], 5., delta=.005)
            self.assertTrue(result.info["numerical_converged"])
            self.assertLess(result.info["dimer_curvature_ev_per_a2"], 0)
            self.assertFalse(result.info["saddle_order_verified"])
            self.assertFalse((root / "vasp/WAVECAR").exists())
            frames = read_restart(root / "trajectory.extxyz", index=":")
            for frame in frames:
                x = frame.positions[1, 0] - 5
                self.assertAlmostEqual(frame.info["evaluated_x"], frame.positions[1, 0], places=12)
                self.assertAlmostEqual(frame.get_potential_energy(), (x*x-1)**2, places=11)
                self.assertAlmostEqual(frame.get_forces(apply_constraint=False)[1, 0], -4*x*(x*x-1), places=10)
                self.assertEqual(frame.get_forces(apply_constraint=False)[0, 0], 2.)
                self.assertEqual(frame.info["fermi_ev"], -4.6)
                self.assertIn("dimer_forces", frame.arrays)
                np.testing.assert_allclose(frame.arrays["dimer_mode"][0], 0.)

    def test_dimer_checkpoint_resume_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = dimer_guess(band_fixture())
            with self.assertRaisesRegex(RuntimeError, "Dimer not converged"):
                dimer(seed, WaveToy(root / "vasp"), directory=root, steps=0)
            self.assertTrue((root / "latest.extxyz").exists())
            self.assertTrue((root / "vasp/WAVECAR").exists())
            self.assertFalse((root / "final.extxyz").exists())
            with self.assertRaisesRegex(FileExistsError, "resume"):
                dimer(seed, WaveToy(root / "vasp"), directory=root)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                dimer(seed, WaveToy(root / "vasp"), directory=root, resume=True, dimer_separation=.02)
            with self.assertRaisesRegex(ValueError, "reservoir"):
                dimer(seed, ToyCP(-4.2), directory=root, resume=True)
            result = dimer(seed, WaveToy(root / "vasp"), directory=root, resume=True, keep_wavecar=True)
            self.assertTrue(result.info["numerical_converged"])
            self.assertTrue((root / "vasp/WAVECAR").exists())
            frames = read_restart(root / "trajectory.extxyz", index=":")
            self.assertEqual([f.info["step"] for f in frames], list(range(len(frames))))

    def test_dimer_rotates_the_mode_and_relaxes_transverse_coordinates(self):
        defaults = DimerControl.parameters.copy()
        with tempfile.TemporaryDirectory() as temp:
            atoms = structure(5.15)
            atoms.positions[1, 1:] += [.15, -.1]
            result = dimer(atoms, ToyCP(), directory=temp, mode=[[0, 0, 0], [1, .3, .2]],
                           fmax=.005, steps=150, use_central_forces=False)
            np.testing.assert_allclose(result.positions[1], [5, 5, 8], atol=.006)
            self.assertAlmostEqual(abs(result.arrays["dimer_mode"][1, 0]), 1., delta=.001)
            self.assertEqual(DimerControl.parameters, defaults)

    def test_dimer_probes_run_external_electron_control_and_export_center_charge(self):
        def fake_vasp(calc, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            calc.atoms = atoms.copy()
            calc.sort = calc.resort = np.arange(len(atoms))
            root = Path(calc.directory)
            root.mkdir(parents=True, exist_ok=True)
            n = calc.float_params.get("nelect")
            n = round(11. if n is None else n, 6) - 11.
            x, y, z = atoms.positions[1] - [5, 5, 8]
            mu0 = -4.2 + .3*x
            free = (x*x-1)**2 + .5*(y*y+z*z) + mu0*n + .25*n*n
            forces = np.array([[2., 0., 0.], [-4*x*(x*x-1)-.3*n, -y, -z]])
            (root / "OUTCAR").write_text(fixed_output(11+n, mu0+.5*n, free))
            (root / "POTCAR").write_text("synthetic PAW hash fixture")
            (root / "WAVECAR").write_text("synthetic wavefunction fixture")
            calc.results = dict(energy=free, free_energy=free, forces=forces)
            calc.converged, calc.version = True, "synthetic_backend"
            calc._store_param_state()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calc = CPVasp(-4.2, directory=root / "vasp", gga="RP", pp="PBE", tolerance=1e-5)
            with mock.patch.object(Vasp, "calculate", fake_vasp), \
                    mock.patch.object(CPVasp, "default_nelect_from_ppp", return_value=11.), \
                    redirect_stdout(io.StringIO()):
                result = dimer(structure(5.15), calc, directory=root, mode=[[0, 0, 0], [1, 0, 0]],
                               steps=100, fmax=.005, keep_wavecar=True)
            self.assertTrue(result.info["numerical_converged"])
            evaluations = read_restart(root / "vasp/evaluations.extxyz", index=":")
            self.assertTrue(any(f.info["dimer_evaluation"] == "displaced" for f in evaluations))
            self.assertGreater(np.ptp([f.info["nelect"] for f in evaluations]), .01)
            frames = read_restart(root / "trajectory.extxyz", index=":")
            for frame in frames:
                x = frame.positions[1, 0] - 5
                self.assertTrue(frame.info["potential_converged"])
                # E-fermi is printed to four decimals in this VASP-like fixture.
                self.assertAlmostEqual(frame.info["excess_electrons"], -.6*x, delta=1.1e-4)
                self.assertAlmostEqual(frame.get_potential_energy(), (x*x-1)**2 - .09*x*x, places=7)

    def test_dimer_does_not_accept_a_minimum_as_a_saddle(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "Dimer not converged"):
                dimer(structure(4.), ToyCP(), directory=temp, mode=[[0, 0, 0], [1, 0, 0]], steps=0)
            frame = read_restart(Path(temp) / "latest.extxyz")
            self.assertGreater(frame.info["dimer_curvature_ev_per_a2"], 0)
            self.assertFalse(frame.info["numerical_converged"])

    def test_dimer_probe_failure_preserves_checkpoint_and_cache(self):
        class BrokenProbe(WaveToy):
            def calculate(self, atoms=None, *args, **kwargs):
                if self.calls and atoms.info.get("dimer_evaluation") == "displaced":
                    raise RuntimeError("synthetic failed dimer SCF")
                super().calculate(atoms, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = dimer_guess(band_fixture())
            with self.assertRaisesRegex(RuntimeError, "Dimer not converged"):
                dimer(seed, WaveToy(root / "vasp"), directory=root, steps=0)
            checkpoint = (root / "latest.extxyz").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "failed dimer SCF"):
                dimer(seed, BrokenProbe(root / "vasp"), directory=root, resume=True)
            self.assertEqual((root / "latest.extxyz").read_bytes(), checkpoint)
            self.assertTrue((root / "vasp/WAVECAR").exists())
            self.assertFalse((root / "final.extxyz").exists())


if __name__ == "__main__":
    unittest.main()
