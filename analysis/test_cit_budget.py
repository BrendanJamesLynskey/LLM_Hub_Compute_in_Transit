"""Checks for cit_budget.py.

Run: python -m pytest -q analysis   (needs a checkout of Disaggregated_Inference_Sim installed with pip -e)
"""
import math

import numpy as np
import pytest

import cit_budget as C
from disagg_sim.hardware import LLAMA3_8B


def test_budget_bracket_on_qsfp56():
    assert C.budget(10e9, 25e9) == pytest.approx(0.4)
    assert C.budget(99e9, 25e9) == pytest.approx(3.96)


def test_fft_cost_convention():
    # FFTW convention: real FFT of length n = 2.5 n log2 n FLOPs, so 2.5 log2 n per value
    assert C.fft_per_value(1024) == pytest.approx(25.0)
    assert C.dct2d_per_value(64) == pytest.approx(30.0)


def test_kv_bytes_match_the_fourier_optics_series():
    # FOptInf analysis/results.md section 9: GQA KV cache 131,072 bytes per prompt token
    assert LLAMA3_8B.kv_bytes_per_token == 131072


def test_ring_allreduce_volume():
    assert C.ring_bytes_per_worker(1.0, 2) == pytest.approx(1.0)
    assert C.ring_bytes_per_worker(1.0, 8) == pytest.approx(1.75)


def test_dct_is_orthonormal():
    D = C.dct_matrix(64)
    assert np.allclose(D @ D.T, np.eye(64), atol=1e-12)


def test_topk_never_keeps_less_energy_than_a_band_of_the_same_size():
    for r in C.band_vs_topk(np.random.default_rng(1), chunks=16):
        assert r["topk"] >= r["band"] - 1e-12
    white = C.band_vs_topk(np.random.default_rng(2), chunks=64)[0]
    assert white["band"] == pytest.approx(white["keep"], abs=0.01)   # noise: a band keeps its own share


def test_kernel_rows_are_consistent():
    for k in C.kernels():
        assert k["transform_per_byte"] >= 0 and k["other_per_byte"] >= 0
        assert not (math.isnan(k["transform_per_byte"]) or math.isnan(k["other_per_byte"]))
