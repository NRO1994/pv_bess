"""Tests for portfolio.system_value – World A calculation."""

from __future__ import annotations

import numpy as np
import pytest

from pv_bess_model.portfolio.system_value import WorldAResult, compute_world_a


class TestComputeWorldA:
    """Tests for compute_world_a()."""

    def test_pure_pv_no_load(self) -> None:
        """Pure PV (no load) → all surplus sold, system_cost < 0 (net revenue)."""
        pv = np.full(96, 10.0)   # 10 kWh per interval
        load = np.zeros(96)
        prices = np.full(96, 0.05)  # 5 ct/kWh

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_kwh == pytest.approx(960.0)
        assert result.total_buy_kwh == pytest.approx(0.0)
        assert result.total_sell_eur == pytest.approx(48.0)
        assert result.total_buy_eur == pytest.approx(0.0)
        assert result.system_cost == pytest.approx(-48.0)

    def test_pure_load_no_pv(self) -> None:
        """Pure load (no PV) → all deficit bought, system_cost > 0."""
        pv = np.zeros(96)
        load = np.full(96, 10.0)
        prices = np.full(96, 0.05)

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_kwh == pytest.approx(0.0)
        assert result.total_buy_kwh == pytest.approx(960.0)
        assert result.system_cost == pytest.approx(48.0)

    def test_pv_equals_load(self) -> None:
        """PV = load → no grid interaction, system_cost = 0."""
        pv = np.full(96, 10.0)
        load = np.full(96, 10.0)
        prices = np.full(96, 0.05)

        result = compute_world_a(pv, load, prices)

        assert result.system_cost == pytest.approx(0.0)
        assert result.total_sell_kwh == pytest.approx(0.0)
        assert result.total_buy_kwh == pytest.approx(0.0)

    def test_energy_conservation(self) -> None:
        """Total sell + total buy should equal total absolute net position."""
        rng = np.random.RandomState(42)
        pv = rng.uniform(0, 20, 96)
        load = rng.uniform(0, 20, 96)
        prices = rng.uniform(0.02, 0.10, 96)

        result = compute_world_a(pv, load, prices)

        netto_abs = np.sum(np.abs(pv - load))
        assert result.total_sell_kwh + result.total_buy_kwh == pytest.approx(netto_abs)

    def test_netto_array(self) -> None:
        """Netto array should equal pv - load."""
        pv = np.array([10.0, 5.0, 0.0, 15.0])
        load = np.array([5.0, 10.0, 0.0, 5.0])
        prices = np.array([0.05, 0.05, 0.05, 0.05])

        result = compute_world_a(pv, load, prices)

        np.testing.assert_array_almost_equal(result.netto, [5.0, -5.0, 0.0, 10.0])

    def test_different_length_raises(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_world_a(np.zeros(96), np.zeros(48), np.zeros(96))

    def test_varying_prices(self) -> None:
        """Revenue/cost should weight by price per interval."""
        pv = np.array([10.0, 0.0])       # sell 10 in interval 0
        load = np.array([0.0, 10.0])      # buy 10 in interval 1
        prices = np.array([0.10, 0.02])   # sell@10ct, buy@2ct

        result = compute_world_a(pv, load, prices)

        assert result.total_sell_eur == pytest.approx(1.0)   # 10 × 0.10
        assert result.total_buy_eur == pytest.approx(0.2)    # 10 × 0.02
        assert result.system_cost == pytest.approx(-0.8)     # 0.2 - 1.0

    def test_negative_prices(self) -> None:
        """Negative spot prices: selling at negative price = cost."""
        pv = np.array([10.0])
        load = np.array([0.0])
        prices = np.array([-0.05])

        result = compute_world_a(pv, load, prices)

        # Selling 10 kWh at -0.05 EUR/kWh = -0.5 EUR revenue
        assert result.total_sell_eur == pytest.approx(-0.5)
        # system_cost = 0 - (-0.5) = 0.5 (net cost from negative price)
        assert result.system_cost == pytest.approx(0.5)

    def test_returns_world_a_result(self) -> None:
        """Return type should be WorldAResult."""
        result = compute_world_a(np.zeros(4), np.zeros(4), np.zeros(4))
        assert isinstance(result, WorldAResult)
