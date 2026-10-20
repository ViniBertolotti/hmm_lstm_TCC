import numpy as np
from scipy import stats

from src.risk.density import berkowitz_test, crps_ensemble, ks_uniform, pit_ensemble


def test_crps_of_a_large_ensemble_matches_the_normal_closed_form():
    rng = np.random.default_rng(0)
    sigma, y = .02, .013
    draws = rng.normal(0, sigma, 200_000)
    z = y / sigma
    exact = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert abs(crps_ensemble(draws, y) - exact) < 2e-4 * sigma / .02
    assert crps_ensemble([y], y) == 0  # previsão pontual certa: CRPS zero


def test_pit_is_uniform_when_the_forecast_is_right_and_not_when_too_narrow():
    rng = np.random.default_rng(1)
    # Cenários novos a cada dia (um conjunto fixo teria erro amostral comum a todos os dias).
    right = [pit_ensemble(rng.normal(0, 1, 252), rng.normal(), rng) for _ in range(3000)]
    narrow = [pit_ensemble(rng.normal(0, .5, 252), rng.normal(), rng) for _ in range(3000)]
    assert ks_uniform(right)["p_value"] > .01 and berkowitz_test(right)["p_value"] > .01
    assert ks_uniform(narrow)["p_value"] < 1e-6 and berkowitz_test(narrow)["p_value"] < 1e-6
    assert berkowitz_test(narrow)["sigma"] > 1.5  # caudas subestimadas: z disperso demais
