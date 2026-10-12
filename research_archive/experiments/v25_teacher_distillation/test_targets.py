#!/usr/bin/env python3
import numpy as np

from experiments.v25_teacher_distillation.adapt_teacher_target import blended_targets


def test_blended_targets_are_normalized():
    raw = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=np.float32)
    structured = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    targets = blended_targets(raw, structured, structure_weight=0.20, smoothing=0.02)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(targets > 0)
    assert targets[0].argmax() == 0
    assert targets[1].argmax() == 2


def test_structure_endpoints():
    raw = np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32)
    structured = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    raw_only = blended_targets(raw, structured, structure_weight=0.0, smoothing=0.0)
    hard_only = blended_targets(raw, structured, structure_weight=1.0, smoothing=0.0)
    np.testing.assert_allclose(raw_only, raw, atol=1e-6)
    np.testing.assert_allclose(hard_only, structured, atol=1e-6)
