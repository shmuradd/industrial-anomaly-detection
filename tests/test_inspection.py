import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from inspection.data import split_normal
from inspection.model import Encoder, calibrate, fit_statistics, image_scores, score_maps


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.features = self.rng.normal(size=(30, 4, 8, 8)).astype(np.float32)

    def test_constant_channels_stay_finite(self):
        features = np.zeros((3, 2, 4, 4), dtype=np.float32)
        mean, variance = fit_statistics(features)
        self.assertTrue((variance > 0).all())
        self.assertTrue(np.isfinite(score_maps(features, mean, variance)).all())

    def test_local_defect_increases_score(self):
        mean, variance = fit_statistics(self.features)
        normal = mean[None].copy()
        defect = normal.copy()
        defect[:, :, 3:5, 3:5] += 8
        normal_map = score_maps(normal, mean, variance)
        defect_map = score_maps(defect, mean, variance)
        self.assertGreater(image_scores(defect_map)[0], image_scores(normal_map)[0] + 1)
        self.assertGreater(defect_map[0, 128, 128], defect_map[0, 0, 0])

    def test_calibration_finite_sample_rank(self):
        self.assertEqual(calibrate(np.arange(42)), 40)
        # Equality is accepted; only a score above this bound is flagged
        self.assertFalse(40 > calibrate(np.arange(42)))

    def test_calibration_rejects_insufficient_data(self):
        with self.assertRaises(ValueError):
            calibrate(np.arange(5))
        with self.assertRaises(ValueError):
            calibrate([np.nan] * 42)

    def test_invalid_statistics(self):
        with self.assertRaises(ValueError):
            fit_statistics(self.features[:1])
        with self.assertRaises(ValueError):
            score_maps(self.features, np.zeros((4, 8, 8)), np.zeros((4, 8, 8)))

    def test_split_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train/good").mkdir(parents=True)
            for i in range(209):
                (root / "train/good" / f"{i:03}.png").touch()
            fitting, calibration = split_normal(root)
            self.assertEqual((len(fitting), len(calibration)), (167, 42))
            self.assertFalse(set(fitting) & set(calibration))
            self.assertEqual((fitting, calibration), split_normal(root))

    def test_encoder_shape_and_frozen_parameters(self):
        torch.set_num_threads(2)
        encoder = Encoder(pretrained=False)
        with torch.inference_mode():
            output = encoder(torch.zeros(1, 3, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 192, 32, 32))
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
