"""Tests for finance/metrics.py – IRR, NPV, DSCR, LCOE, Payback Period.

Reference IRR: cashflows [-1000, 300, 300, 300, 300]
  Solve: 300 × [(1-(1+r)^-4) / r] = 1000  →  r ≈ 7.71 %
"""

from __future__ import annotations

import math

import numpy as np
import numpy_financial as npf
import pytest

from pv_bess_model.finance.metrics import (
    calculate_dscr,
    calculate_lcoe,
    calculate_npv,
    calculate_payback_year,
    compute_all_metrics,
    safe_irr,
)


# ---------------------------------------------------------------------------
# safe_irr
# ---------------------------------------------------------------------------


class TestSafeIrr:
    """Tests for safe_irr."""

    def test_known_irr(self) -> None:
        """[-1000, 300, 300, 300, 300] → IRR ≈ 7.71 %."""
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0])
        result = safe_irr(cf)
        assert result is not None
        # Cross-check with numpy_financial
        expected = float(npf.irr(cf))
        assert math.isclose(result, expected, rel_tol=1e-6)
        assert math.isclose(result, 0.0771, abs_tol=1e-4)

    def test_negative_irr(self) -> None:
        """Investment that loses money: [-1000, 100, 100, 100] → negative IRR."""
        cf = np.array([-1000.0, 100.0, 100.0, 100.0])
        result = safe_irr(cf)
        assert result is not None
        assert result < 0.0

    def test_zero_cashflows_returns_none(self) -> None:
        """All-zero cashflows → IRR should return None (non-convergent)."""
        cf = np.array([0.0, 0.0, 0.0])
        result = safe_irr(cf)
        assert result is None

    def test_all_positive_returns_none(self) -> None:
        """All-positive cashflows → no sign change, IRR not defined."""
        cf = np.array([100.0, 200.0, 300.0])
        result = safe_irr(cf)
        # numpy_financial.irr returns nan for no sign change
        assert result is None

    def test_exact_breakeven(self) -> None:
        """[-1000, 500, 500] → IRR = 0.0 (exact breakeven)."""
        cf = np.array([-1000.0, 500.0, 500.0])
        result = safe_irr(cf)
        assert result is not None
        assert math.isclose(result, 0.0, abs_tol=1e-6)

    def test_high_return(self) -> None:
        """[-100, 200] → IRR = 100 %."""
        cf = np.array([-100.0, 200.0])
        result = safe_irr(cf)
        assert result is not None
        assert math.isclose(result, 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# calculate_npv
# ---------------------------------------------------------------------------


class TestCalculateNpv:
    """Tests for calculate_npv."""

    def test_npv_at_irr_is_zero(self) -> None:
        """NPV at the IRR discount rate must be ~0."""
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0])
        irr = safe_irr(cf)
        assert irr is not None
        npv_at_irr = calculate_npv(cf, discount_rate=irr)
        assert math.isclose(npv_at_irr, 0.0, abs_tol=1e-4)

    def test_npv_zero_discount(self) -> None:
        """At 0 % discount, NPV = sum of all cashflows."""
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0])
        result = calculate_npv(cf, discount_rate=0.0)
        assert math.isclose(result, 200.0, abs_tol=1e-6)

    def test_npv_reference_at_six_percent(self) -> None:
        """NPV at 6 % cross-checked with numpy_financial.npv."""
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0])
        result = calculate_npv(cf, discount_rate=0.06)
        expected = float(npf.npv(0.06, cf))
        assert math.isclose(result, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# calculate_dscr
# ---------------------------------------------------------------------------


class TestCalculateDscr:
    """Tests for calculate_dscr."""

    def test_uniform_dscr(self) -> None:
        """Constant revenue/opex/debt → min = avg."""
        revenues = [500_000.0] * 5
        opex = [200_000.0] * 5
        debt_svc = [100_000.0] * 5
        min_d, avg_d = calculate_dscr(revenues, opex, debt_svc)
        assert min_d is not None and avg_d is not None
        expected = (500_000 - 200_000) / 100_000  # 3.0
        assert math.isclose(min_d, expected)
        assert math.isclose(avg_d, expected)

    def test_varying_dscr(self) -> None:
        """Check min and avg with varying revenues."""
        revenues = [300_000.0, 400_000.0, 500_000.0]
        opex = [200_000.0, 200_000.0, 200_000.0]
        debt_svc = [100_000.0, 100_000.0, 100_000.0]
        min_d, avg_d = calculate_dscr(revenues, opex, debt_svc)
        assert min_d is not None and avg_d is not None
        # DSCRs: 1.0, 2.0, 3.0
        assert math.isclose(min_d, 1.0)
        assert math.isclose(avg_d, 2.0)

    def test_no_debt_returns_none(self) -> None:
        """No debt service → (None, None)."""
        min_d, avg_d = calculate_dscr([100_000.0], [50_000.0], [0.0])
        assert min_d is None
        assert avg_d is None

    def test_empty_lists_returns_none(self) -> None:
        """Empty lists → (None, None)."""
        min_d, avg_d = calculate_dscr([], [], [])
        assert min_d is None
        assert avg_d is None

    def test_dscr_below_one_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """DSCR < 1 should log a warning."""
        revenues = [80_000.0]
        opex = [50_000.0]
        debt_svc = [100_000.0]
        with caplog.at_level("WARNING"):
            min_d, _ = calculate_dscr(revenues, opex, debt_svc)
        assert min_d is not None
        assert min_d < 1.0
        assert "below threshold" in caplog.text


# ---------------------------------------------------------------------------
# calculate_lcoe
# ---------------------------------------------------------------------------


class TestCalculateLcoe:
    """Tests for calculate_lcoe."""

    def test_simple_lcoe(self) -> None:
        """LCOE = total costs / total production."""
        result = calculate_lcoe(10_000_000.0, 100_000_000.0)
        assert result is not None
        assert math.isclose(result, 0.10)  # 0.10 €/kWh

    def test_zero_production_returns_none(self) -> None:
        """Zero production → LCOE is undefined."""
        result = calculate_lcoe(10_000_000.0, 0.0)
        assert result is None

    def test_negative_production_returns_none(self) -> None:
        """Negative production → LCOE is undefined."""
        result = calculate_lcoe(10_000_000.0, -1.0)
        assert result is None


# ---------------------------------------------------------------------------
# calculate_payback_year
# ---------------------------------------------------------------------------


class TestCalculatePaybackYear:
    """Tests for calculate_payback_year."""

    def test_payback_in_year_4(self) -> None:
        """[-1000, 300, 300, 300, 300] → cumulative turns positive in year 4."""
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0])
        # Cumulative: -1000, -700, -400, -100, +200
        result = calculate_payback_year(cf)
        assert result == 4

    def test_immediate_payback(self) -> None:
        """Positive initial CF → payback at year 0."""
        cf = np.array([100.0, 50.0])
        result = calculate_payback_year(cf)
        assert result == 0

    def test_no_payback(self) -> None:
        """Never turns positive → None."""
        cf = np.array([-1000.0, 100.0, 100.0])
        result = calculate_payback_year(cf)
        assert result is None

    def test_exact_breakeven_not_payback(self) -> None:
        """Cumulative exactly 0 is not > 0 → no payback."""
        cf = np.array([-100.0, 50.0, 50.0])
        # Cumulative: -100, -50, 0
        result = calculate_payback_year(cf)
        assert result is None


# ---------------------------------------------------------------------------
# compute_all_metrics (integration)
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    """Integration test for compute_all_metrics."""

    def test_all_metrics_returned(self) -> None:
        """Smoke test: all fields are populated for a valid scenario."""
        equity_cf = np.array([-500_000.0] + [80_000.0] * 10)
        project_cf = np.array([-1_000_000.0] + [120_000.0] * 10)
        revenues = [200_000.0] * 10
        opex = [80_000.0] * 10
        debt_svc = [40_000.0] * 10

        m = compute_all_metrics(
            equity_cashflows=equity_cf,
            project_cashflows=project_cf,
            annual_revenues=revenues,
            annual_opex=opex,
            annual_debt_service=debt_svc,
            total_capex=1_000_000.0,
            total_opex_lifetime=800_000.0,
            total_production_kwh=5_000_000.0,
            discount_rate=0.06,
        )
        assert m.equity_irr is not None
        assert m.project_irr is not None
        assert isinstance(m.npv, float)
        assert m.dscr_min is not None
        assert m.dscr_avg is not None
        assert m.lcoe is not None
        assert m.payback_year is not None

    def test_lcoe_calculation(self) -> None:
        """LCOE = (CAPEX + lifetime OPEX) / total production."""
        equity_cf = np.array([-100.0, 50.0, 50.0, 50.0])
        project_cf = np.array([-100.0, 50.0, 50.0, 50.0])

        m = compute_all_metrics(
            equity_cashflows=equity_cf,
            project_cashflows=project_cf,
            annual_revenues=[50.0] * 3,
            annual_opex=[10.0] * 3,
            annual_debt_service=[0.0] * 3,
            total_capex=1_000.0,
            total_opex_lifetime=500.0,
            total_production_kwh=10_000.0,
            discount_rate=0.06,
        )
        assert m.lcoe is not None
        assert math.isclose(m.lcoe, (1_000.0 + 500.0) / 10_000.0)
