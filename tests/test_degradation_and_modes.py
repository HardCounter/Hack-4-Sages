"""Smoke tests for graceful degradation and LLM mode selection."""

import numpy as np
import pytest


class TestGracefulDegradation:
    def test_fallback_invoked_on_failure(self):
        from modules.degradation import GracefulDegradation

        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("boom")

        def fallback():
            return "ok"

        result = gd.run_with_fallback(failing, fallback, timeout=2.0, label="test")
        assert result == "ok"

    def test_validate_temperature_map_rejects_nan(self):
        from modules.degradation import GracefulDegradation

        gd = GracefulDegradation()
        bad = np.full((4, 4), np.nan)
        assert gd.validate_temperature_map(bad) is False

    def test_validate_temperature_map_accepts_good(self):
        from modules.degradation import GracefulDegradation

        gd = GracefulDegradation()
        good = np.full((4, 4), 300.0)
        assert gd.validate_temperature_map(good) is True


class TestAgentModes:
    def test_agent_mode_enum(self):
        pytest.importorskip("langchain")
        from modules.agent_setup import AgentMode

        assert AgentMode.DUAL_LLM.value == "dual_llm"
        assert AgentMode.SINGLE_LLM.value == "single_llm"
        assert AgentMode.DETERMINISTIC.value == "deterministic"

    def test_deterministic_returns_none(self):
        pytest.importorskip("langchain")
        from modules.agent_setup import AgentMode, build_agent

        agent = build_agent(AgentMode.DETERMINISTIC)
        assert agent is None


class TestLLMHelpersModeSwitch:
    def test_set_llm_mode_accepted(self):
        pytest.importorskip("ollama")
        from modules.llm_helpers import set_llm_mode

        set_llm_mode("single_llm")
        set_llm_mode("dual_llm")
        set_llm_mode("deterministic")


class TestGCMBenchmarks:
    def test_list_benchmarks(self):
        from modules.gcm_benchmarks import list_benchmarks

        keys = list_benchmarks()
        assert "earth_like" in keys
        assert "proxima_b" in keys
        assert "hot_rock" in keys

    def test_get_benchmark_returns_map(self):
        from modules.gcm_benchmarks import get_gcm_benchmark

        bench = get_gcm_benchmark("earth_like")
        assert bench is not None
        assert "temperature_map" in bench
        assert bench["temperature_map"].shape == (32, 64)

    def test_compare_surrogate_to_gcm(self):
        from modules.gcm_benchmarks import compare_surrogate_to_gcm

        a = np.random.RandomState(0).normal(280, 10, (32, 64))
        b = np.random.RandomState(1).normal(280, 10, (32, 64))
        metrics = compare_surrogate_to_gcm(a, b)
        assert "pattern_correlation" in metrics
        assert "rmse_K" in metrics
        assert "bias_K" in metrics
