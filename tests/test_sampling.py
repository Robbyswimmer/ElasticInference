"""Tests for the sampling logic used in prefill/decode workers."""
import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "proto"))
import inference_pb2

from workers.prefill import _apply_sampling


class TestGreedySampling:

    def test_argmax_when_no_params(self):
        """With all-zero sampling params, workers use argmax (not _apply_sampling)."""
        logits = torch.zeros(100)
        logits[42] = 10.0
        sp = inference_pb2.SamplingParams()
        # Simulate the worker's greedy path
        assert sp.temperature == 0.0
        assert torch.argmax(logits).item() == 42


class TestTemperature:

    def test_high_temperature_increases_entropy(self):
        """Higher temperature should produce more varied outputs."""
        torch.manual_seed(0)
        logits = torch.zeros(1000)
        logits[0] = 5.0  # One dominant token
        sp_low = inference_pb2.SamplingParams(temperature=0.1)
        sp_high = inference_pb2.SamplingParams(temperature=2.0)

        low_results = set()
        high_results = set()
        for _ in range(50):
            low_results.add(_apply_sampling(logits.clone(), sp_low))
            high_results.add(_apply_sampling(logits.clone(), sp_high))

        # High temp should produce more unique tokens
        assert len(high_results) >= len(low_results)

    def test_very_low_temperature_is_near_greedy(self):
        logits = torch.zeros(100)
        logits[7] = 10.0
        sp = inference_pb2.SamplingParams(temperature=0.01)
        results = [_apply_sampling(logits.clone(), sp) for _ in range(20)]
        # Almost always picks token 7
        assert results.count(7) >= 18


class TestTopK:

    def test_limits_to_k_tokens(self):
        """Top-k should only sample from the k highest logits."""
        torch.manual_seed(42)
        logits = torch.arange(100, dtype=torch.float)  # 0..99, token 99 highest
        sp = inference_pb2.SamplingParams(top_k=5, temperature=1.0)
        results = set(_apply_sampling(logits.clone(), sp) for _ in range(100))
        # All results should be in top-5: tokens 95-99
        assert all(r >= 95 for r in results)

    def test_top_k_1_is_greedy(self):
        logits = torch.randn(500)
        sp = inference_pb2.SamplingParams(top_k=1, temperature=1.0)
        expected = torch.argmax(logits).item()
        for _ in range(10):
            assert _apply_sampling(logits.clone(), sp) == expected


class TestTopP:

    def test_nucleus_excludes_low_prob_tokens(self):
        """Top-p should exclude tokens outside the nucleus."""
        torch.manual_seed(0)
        logits = torch.zeros(1000)
        logits[0] = 10.0  # Dominates probability mass
        logits[1] = 9.0
        # Rest are 0.0 — very low probability
        sp = inference_pb2.SamplingParams(top_p=0.5, temperature=1.0)
        results = set(_apply_sampling(logits.clone(), sp) for _ in range(50))
        # Should mostly pick from tokens 0 and 1
        assert len(results) <= 5


class TestRepetitionPenalty:

    def test_penalizes_seen_tokens(self):
        """Repetition penalty should reduce probability of already-generated tokens."""
        torch.manual_seed(0)
        logits = torch.zeros(100)
        logits[10] = 5.0
        logits[20] = 4.9  # Close second

        sp_no_penalty = inference_pb2.SamplingParams(temperature=0.01)
        sp_penalty = inference_pb2.SamplingParams(temperature=0.01, repetition_penalty=2.0)

        # Without penalty, token 10 dominates
        no_pen = [_apply_sampling(logits.clone(), sp_no_penalty) for _ in range(10)]
        assert no_pen.count(10) >= 9

        # With penalty on token 10, token 20 should appear more
        with_pen = [_apply_sampling(logits.clone(), sp_penalty, generated_ids=[10]) for _ in range(10)]
        assert with_pen.count(20) >= with_pen.count(10)

    def test_no_penalty_when_1(self):
        logits = torch.zeros(50)
        logits[5] = 10.0
        sp = inference_pb2.SamplingParams(temperature=0.01, repetition_penalty=1.0)
        results = [_apply_sampling(logits.clone(), sp, generated_ids=[5]) for _ in range(10)]
        assert results.count(5) >= 9


class TestEdgeCases:

    def test_single_token_vocab(self):
        logits = torch.tensor([1.0])
        sp = inference_pb2.SamplingParams(temperature=1.0)
        assert _apply_sampling(logits.clone(), sp) == 0

    def test_returns_valid_range(self):
        torch.manual_seed(0)
        vocab_size = 50257
        logits = torch.randn(vocab_size)
        sp = inference_pb2.SamplingParams(temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1)
        for _ in range(20):
            token = _apply_sampling(logits.clone(), sp, [1, 2, 3])
            assert 0 <= token < vocab_size
