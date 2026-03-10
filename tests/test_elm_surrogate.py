"""Tests for the ELM climate surrogate."""

import numpy as np
import pytest

from modules.elm_surrogate import (
    ELMClimateSurrogate,
    ELMEnsemble,
    PureELM,
    generate_analytical_training_data,
)


class TestPureELM:
    def test_fit_predict_shape(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 10)
        elm = PureELM(n_neurons=50)
        elm.fit(X, y)
        pred = elm.predict(X)
        assert pred.shape == y.shape

    def test_predict_not_constant(self):
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3, 5) + 0.1 * np.random.randn(50, 5)
        elm = PureELM(n_neurons=100)
        elm.fit(X, y)
        pred = elm.predict(X)
        assert pred.std() > 0.01


class TestELMEnsemble:
    def test_ensemble_reduces_variance(self):
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 8)
        ens = ELMEnsemble(K=5, n_neurons=50)
        ens.fit(X, y)
        std = ens.predict_std(X)
        assert std.mean() >= 0


class TestAnalyticalData:
    def test_shape(self):
        X, y = generate_analytical_training_data(n_samples=20, n_lat=8, n_lon=16)
        assert X.shape == (20, 8)
        assert y.shape == (20, 8 * 16)

    def test_temperatures_positive(self):
        _, y = generate_analytical_training_data(n_samples=50)
        assert np.all(y > 0)


class TestELMClimateSurrogate:
    def test_train_predict_round_trip(self):
        X, y = generate_analytical_training_data(n_samples=100, n_lat=8, n_lon=16)
        model = ELMClimateSurrogate(n_ensemble=3, n_neurons=50)
        model.N_LAT = 8
        model.N_LON = 16
        model.train(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (8, 16)
        assert np.all(np.isfinite(pred))

    def test_conformal_intervals(self):
        X, y = generate_analytical_training_data(n_samples=100, n_lat=8, n_lon=16)
        model = ELMClimateSurrogate(n_ensemble=5, n_neurons=50)
        model.N_LAT = 8
        model.N_LON = 16
        model.train(X, y)
        mean, lower, upper = model.predict_conformal(X[:1], alpha=0.1)
        assert mean.shape == (8, 16)
        assert np.all(lower <= mean)
        assert np.all(upper >= mean)
