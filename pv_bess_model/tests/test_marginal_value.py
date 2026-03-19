"""Tests for portfolio.marginal_value – Marginal value curves."""

from __future__ import annotations

import pytest

from pv_bess_model.portfolio.marginal_value import (
    MarginalValuePoint,
    compute_marginal_values,
)
from pv_bess_model.portfolio.system_value import SystemValuePoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_point(
    rate: float,
    value: float,
    etp: float = 2.0,
    name: str = "BESS_1",
) -> SystemValuePoint:
    """Helper to create a SystemValuePoint."""
    return SystemValuePoint(
        flex_name=name,
        flex_type="bess",
        annual_addition_kw=rate,
        e_to_p_ratio=etp,
        cumulative_system_value_eur=value,
        annual_system_values=[value / 25] * 25,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeMarginalValues:
    """Tests for the marginal value computation."""

    def test_empty_input(self) -> None:
        """Empty input produces empty output."""
        result = compute_marginal_values([])
        assert result == []

    def test_single_point(self) -> None:
        """Single point: marginal = value / rate."""
        points = [_make_point(rate=100.0, value=50000.0)]
        result = compute_marginal_values(points)
        assert len(result) == 1
        assert result[0].marginal_value_eur_per_kw_a == pytest.approx(500.0)
        assert result[0].delta_kw == pytest.approx(100.0)
        assert result[0].delta_value_eur == pytest.approx(50000.0)

    def test_two_points_linear(self) -> None:
        """Two points with linear system value."""
        points = [
            _make_point(rate=100.0, value=50000.0),
            _make_point(rate=200.0, value=100000.0),
        ]
        result = compute_marginal_values(points)
        assert len(result) == 2
        # First: 50000 / 100 = 500
        assert result[0].marginal_value_eur_per_kw_a == pytest.approx(500.0)
        # Second: (100000 - 50000) / (200 - 100) = 500
        assert result[1].marginal_value_eur_per_kw_a == pytest.approx(500.0)

    def test_diminishing_returns(self) -> None:
        """Marginal value should decrease with diminishing returns."""
        points = [
            _make_point(rate=100.0, value=50000.0),
            _make_point(rate=200.0, value=80000.0),
            _make_point(rate=300.0, value=100000.0),
        ]
        result = compute_marginal_values(points)
        assert len(result) == 3
        # 50000/100=500, 30000/100=300, 20000/100=200
        assert result[0].marginal_value_eur_per_kw_a == pytest.approx(500.0)
        assert result[1].marginal_value_eur_per_kw_a == pytest.approx(300.0)
        assert result[2].marginal_value_eur_per_kw_a == pytest.approx(200.0)
        # Monotonically decreasing
        for i in range(1, len(result)):
            assert result[i].marginal_value_eur_per_kw_a <= result[i - 1].marginal_value_eur_per_kw_a

    def test_groups_by_etp(self) -> None:
        """Points with different E/P ratios are grouped separately."""
        points = [
            _make_point(rate=100.0, value=30000.0, etp=1.0),
            _make_point(rate=100.0, value=50000.0, etp=2.0),
            _make_point(rate=200.0, value=50000.0, etp=1.0),
            _make_point(rate=200.0, value=80000.0, etp=2.0),
        ]
        result = compute_marginal_values(points)
        assert len(result) == 4

        # Group by etp
        etp1 = [r for r in result if r.e_to_p_ratio == 1.0]
        etp2 = [r for r in result if r.e_to_p_ratio == 2.0]

        assert len(etp1) == 2
        assert len(etp2) == 2

        # etp=1: marginals are 300, 200
        assert etp1[0].marginal_value_eur_per_kw_a == pytest.approx(300.0)
        assert etp1[1].marginal_value_eur_per_kw_a == pytest.approx(200.0)

        # etp=2: marginals are 500, 300
        assert etp2[0].marginal_value_eur_per_kw_a == pytest.approx(500.0)
        assert etp2[1].marginal_value_eur_per_kw_a == pytest.approx(300.0)

    def test_groups_by_name(self) -> None:
        """Points from different flex instances are grouped separately."""
        points = [
            _make_point(rate=100.0, value=40000.0, name="BESS_A"),
            _make_point(rate=100.0, value=30000.0, name="BESS_B"),
        ]
        result = compute_marginal_values(points)
        assert len(result) == 2

        a = [r for r in result if r.flex_name == "BESS_A"]
        b = [r for r in result if r.flex_name == "BESS_B"]

        assert len(a) == 1
        assert len(b) == 1
        assert a[0].marginal_value_eur_per_kw_a == pytest.approx(400.0)
        assert b[0].marginal_value_eur_per_kw_a == pytest.approx(300.0)

    def test_unsorted_input_is_sorted(self) -> None:
        """Points given in arbitrary order are sorted by addition rate."""
        points = [
            _make_point(rate=300.0, value=100000.0),
            _make_point(rate=100.0, value=50000.0),
            _make_point(rate=200.0, value=80000.0),
        ]
        result = compute_marginal_values(points)
        rates = [r.annual_addition_kw for r in result]
        assert rates == [100.0, 200.0, 300.0]

    def test_preserves_cumulative_value(self) -> None:
        """Marginal value points carry the original cumulative value."""
        points = [
            _make_point(rate=50.0, value=10000.0),
            _make_point(rate=100.0, value=25000.0),
        ]
        result = compute_marginal_values(points)
        assert result[0].cumulative_system_value_eur == pytest.approx(10000.0)
        assert result[1].cumulative_system_value_eur == pytest.approx(25000.0)

    def test_zero_rate_point(self) -> None:
        """A point with rate=0 has marginal_value=0."""
        points = [
            _make_point(rate=0.0, value=0.0),
            _make_point(rate=100.0, value=50000.0),
        ]
        result = compute_marginal_values(points)
        assert result[0].marginal_value_eur_per_kw_a == 0.0
        assert result[1].marginal_value_eur_per_kw_a == pytest.approx(500.0)
