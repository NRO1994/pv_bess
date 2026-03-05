"""Chart generation for the PDF report.

Creates matplotlib-based PNG charts for PV yield, price scenarios,
grid search results, and sensitivity analyses.

Public API
----------
create_pv_yield_chart        -- Monthly PV production per weather year.
create_price_scenario_chart  -- Annual mean prices per scenario over lifetime.
create_grid_search_chart     -- Project IRR vs BESS scale per E/P ratio.
create_eeg_sensitivity_chart -- Project IRR vs EEG floor price.
create_ppa_collar_chart      -- Project IRR vs floor price per cap spread.
create_ppa_baseload_chart    -- Project IRR vs PPA price per baseload level.
create_all_charts            -- Convenience wrapper creating all applicable charts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from pv_bess_model.config.defaults import (
    REPORT_CHART_DPI,
    REPORT_CHART_HEIGHT_INCHES,
    REPORT_CHART_WIDTH_INCHES,
    REPORT_CORPORATE_COLORS,
)
from pv_bess_model.config.loader import PriceWeatherScenario
from pv_bess_model.optimization.grid_search import GridSearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_corporate_style(
    fig: Any,
    ax: Any,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: list[str] | None = None,
) -> None:
    """Apply consistent corporate styling to a matplotlib figure/axes.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    ax:
        Matplotlib axes.
    title:
        Chart title.
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.
    colors:
        Corporate color palette (defaults to ``REPORT_CORPORATE_COLORS``).
    """
    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    title_color = colors[3]  # #00467A
    ax.set_title(title, fontsize=14, fontweight="bold", color=title_color, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=title_color)
    ax.set_ylabel(ylabel, fontsize=11, color=title_color)
    ax.tick_params(colors=title_color, labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color(title_color)
        spine.set_linewidth(0.5)
    fig.tight_layout()


def _save_chart(fig: Any, output_path: Path) -> Path:
    """Save a matplotlib figure as PNG and close it.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    output_path:
        Target file path for the PNG.

    Returns
    -------
    Path
        The path the chart was saved to.
    """
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=REPORT_CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------


def create_pv_yield_chart(
    weather_timeseries: dict[int, np.ndarray],
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create a monthly PV production chart with one line per weather year.

    Parameters
    ----------
    weather_timeseries:
        Mapping ``{weather_year: hourly_or_15min_array}``.
        Arrays may have 8760 (hourly) or 35040 (15-min) elements.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    # Days per month for grouping
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for i, (year, ts) in enumerate(sorted(weather_timeseries.items())):
        # Determine intervals per hour
        n = len(ts)
        if n >= 35040:
            intervals_per_hour = 4
        else:
            intervals_per_hour = 1

        monthly_kwh: list[float] = []
        offset = 0
        for days in days_per_month:
            hours = days * 24
            intervals = hours * intervals_per_hour
            chunk = ts[offset : offset + intervals]
            # Each interval is (1/intervals_per_hour) hours
            monthly_kwh.append(float(np.sum(chunk)) / intervals_per_hour/1e6)
            offset += intervals

        color = colors[i % len(colors)]
        ax.plot(range(1, 13), monthly_kwh, marker="o", markersize=4, color=color, label=str(year))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    )
    ax.legend(fontsize=8)
    _apply_corporate_style(fig, ax, "Brutto PV-Ertrag nach Wetterjahr", "Monat", "GWh", colors)
    return _save_chart(fig, output_path)


def create_price_scenario_chart(
    scenario_prices: list[PriceWeatherScenario],
    commissioning_year: int,
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create an annual mean price chart per scenario over project lifetime.

    Parameters
    ----------
    scenario_prices:
        Mapping ``{scenario_name: [year1_array, year2_array, ...]}``.
        Each array contains hourly or 15-min spot prices in EUR/kWh.
    commissioning_year:
        First calendar year of the project.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    for scenario in scenario_prices:
        years = [commissioning_year + y for y in range(len(scenario.price_per_year))]
        # Convert EUR/kWh to EUR/MWh for display
        means = [float(np.mean(arr)) * 1000.0 for arr in scenario.price_per_year]
        label = f"{scenario.name} (WY:{scenario.weather_year})"
        color = colors[scenario.weather_year % len(colors)]
        ax.plot(years, means, marker=".", markersize=3, color=color, label=label)

    ax.legend(fontsize=8)
    _apply_corporate_style(
        fig, ax, "Preisszenarien (Jahresmittel)", "Jahr", "EUR/MWh", colors
    )
    ax.set_xlim(commissioning_year, commissioning_year + len(scenario_prices[0].price_per_year) - 1)
    return _save_chart(fig, output_path)


def create_grid_search_chart(
    grid_result: Any,
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create a grid search chart: Project IRR vs BESS scale per E/P ratio.

    Parameters
    ----------
    grid_result:
        ``GridSearchResult`` instance from the grid search.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    # Group points by e_to_p_ratio
    groups: dict[float, list] = {}
    for pt in grid_result.points:
        ep = pt.e_to_p_ratio
        groups.setdefault(ep, []).append(pt)

    for i, (ep, pts) in enumerate(sorted(groups.items())):
        pts_sorted = sorted(pts, key=lambda p: p.scale_pct)
        scales = [p.scale_pct for p in pts_sorted]
        irrs = [(p.metrics.project_irr or 0.0) * 100.0 for p in pts_sorted]
        color = colors[i % len(colors)]
        ax.plot(scales, irrs, marker="o", markersize=5, color=color, label=f"E/P = {ep:.0f}h")

    # Mark optimal point
    if grid_result.optimal is not None:
        opt = grid_result.optimal
        ax.plot(
            opt.scale_pct,
            (opt.metrics.project_irr or 0.0) * 100.0,
            marker="*",
            markersize=18,
            color=colors[1],
            zorder=5,
            label=f"Optimum ({opt.scale_pct:.0f}%, {opt.e_to_p_ratio:.0f}h)",
        )

    ax.legend(fontsize=8)
    _apply_corporate_style(
        fig, ax, "Grid Search: Project IRR vs. BESS-Skalierung", "BESS-Anteil (% PV)", "Project IRR (%)", colors
    )
    return _save_chart(fig, output_path)


def create_eeg_sensitivity_chart(
    eeg_result: Any,
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create an EEG sensitivity chart: Project IRR vs floor price.

    Parameters
    ----------
    eeg_result:
        ``SensitivityResult`` from the EEG sensitivity analysis.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    floor_prices: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    for point in eeg_result.points:
        fp = point.params.get("floor_price_eur_per_kwh", 0.0) * 100.0  # ct/kWh
        eq_stats = point.mc_result.overall_stats.get("project_irr")
        if eq_stats is not None:
            floor_prices.append(fp)
            means.append(eq_stats.mean * 100.0)
            stds.append(eq_stats.std * 100.0)

    means_arr = np.array(means)
    stds_arr = np.array(stds)

    ax.plot(floor_prices, means, marker="o", markersize=5, color=colors[0], label="Mittelwert")
    ax.fill_between(
        floor_prices,
        means_arr - stds_arr,
        means_arr + stds_arr,
        alpha=0.2,
        color=colors[0],
        label="± 1 Std.Abw.",
    )

    ax.legend(fontsize=8)
    _apply_corporate_style(
        fig, ax, "EEG-Sensitivität: Project IRR vs. Gebotspreis", "EEG-Gebotspreis (ct/kWh)", "Project IRR (%)", colors
    )
    return _save_chart(fig, output_path)


def create_ppa_collar_chart(
    collar_result: Any,
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create a PPA Collar chart: Project IRR vs floor price per cap spread.

    Parameters
    ----------
    collar_result:
        ``SensitivityResult`` from the PPA Collar analysis.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    # Group by cap_spread
    groups: dict[float, list[tuple[float, float]]] = {}
    for point in collar_result.points:
        spread = point.params.get("cap_spread_eur_per_mwh", 0.0)
        floor = point.params.get("floor_price_eur_per_mwh", 0.0)
        eq_stats = point.mc_result.overall_stats.get("project_irr")
        if eq_stats is not None:
            groups.setdefault(spread, []).append((floor, eq_stats.mean * 100.0))

    for i, (spread, data) in enumerate(sorted(groups.items())):
        data_sorted = sorted(data, key=lambda d: d[0])
        floors = [d[0] for d in data_sorted]
        irrs = [d[1] for d in data_sorted]
        color = colors[i % len(colors)]
        ax.plot(floors, irrs, marker="o", markersize=5, color=color, label=f"Cap-Spread = {spread:.0f} EUR/MWh")

    ax.legend(fontsize=8)
    _apply_corporate_style(
        fig, ax, "PPA Collar: Project IRR vs. Floor-Preis", "Floor-Preis (EUR/MWh)", "Project IRR (%)", colors
    )
    return _save_chart(fig, output_path)


def create_ppa_baseload_chart(
    baseload_result: Any,
    output_path: Path,
    colors: list[str] | None = None,
) -> Path:
    """Create a PPA Baseload chart: Project IRR vs PPA price per baseload level.

    Parameters
    ----------
    baseload_result:
        ``SensitivityResult`` from the PPA Baseload analysis.
    output_path:
        File path for the output PNG.
    colors:
        Corporate color palette.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    fig, ax = plt.subplots(
        figsize=(REPORT_CHART_WIDTH_INCHES, REPORT_CHART_HEIGHT_INCHES)
    )

    # Group by baseload_mw
    groups: dict[float, list[tuple[float, float]]] = {}
    for point in baseload_result.points:
        bl = point.params.get("baseload_mw", 0.0)
        ppa_price = point.params.get("ppa_price_eur_per_mwh", 0.0)
        eq_stats = point.mc_result.overall_stats.get("project_irr")
        if eq_stats is not None:
            groups.setdefault(bl, []).append((ppa_price, eq_stats.mean * 100.0))

    for i, (bl, data) in enumerate(sorted(groups.items())):
        data_sorted = sorted(data, key=lambda d: d[0])
        prices = [d[0] for d in data_sorted]
        irrs = [d[1] for d in data_sorted]
        color = colors[i % len(colors)]
        ax.plot(prices, irrs, marker="o", markersize=5, color=color, label=f"Baseload = {bl:.1f} MW")

    ax.legend(fontsize=8)
    _apply_corporate_style(
        fig, ax, "PPA Baseload: Project IRR vs. PPA-Preis", "PPA-Preis (EUR/MWh)", "Project IRR (%)", colors
    )
    return _save_chart(fig, output_path)


def create_all_charts(
    output_dir: Path,
    grid_result: GridSearchResult,
    weather_timeseries: dict[int, np.ndarray] | None = None,
    scenario_prices: list[PriceWeatherScenario] | None = None,
    commissioning_year: int = 2027,
    eeg_result: Any | None = None,
    collar_result: Any | None = None,
    baseload_result: Any | None = None,
    colors: list[str] | None = None,
) -> dict[str, Path]:
    """Create all applicable charts and return a mapping of chart name to path.

    Parameters
    ----------
    output_dir:
        Base output directory. Charts are saved to ``output_dir/charts/``.
    grid_result:
        ``GridSearchResult`` from the grid search (always required).
    weather_timeseries:
        Optional PV weather data for the yield chart.
    scenario_prices:
        Optional price scenario data for the price chart.
    scenario_labels:
        Optional display labels for price scenarios.
    commissioning_year:
        Project commissioning year.
    eeg_result:
        Optional ``SensitivityResult`` from EEG analysis.
    collar_result:
        Optional ``SensitivityResult`` from PPA Collar analysis.
    baseload_result:
        Optional ``SensitivityResult`` from PPA Baseload analysis.
    colors:
        Corporate color palette.

    Returns
    -------
    dict[str, Path]
        Mapping of chart name to the path of the saved PNG file.
    """
    from pv_bess_model.config.defaults import REPORT_CHARTS_SUBDIR

    if colors is None:
        colors = REPORT_CORPORATE_COLORS

    charts_dir = output_dir / REPORT_CHARTS_SUBDIR
    charts_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    # PV yield chart (conditional)
    if weather_timeseries and len(weather_timeseries) > 0:
        try:
            result["pv_yield"] = create_pv_yield_chart(
                weather_timeseries, charts_dir / "pv_yield.png", colors
            )
        except Exception:
            logger.warning("Failed to create PV yield chart.", exc_info=True)

    # Price scenario chart (conditional: need >1 scenario)
    if scenario_prices and len(scenario_prices) > 1:
        try:
            result["price_scenarios"] = create_price_scenario_chart(
                scenario_prices,
                commissioning_year,
                charts_dir / "price_scenarios.png",
                colors,
            )
        except Exception:
            logger.warning("Failed to create price scenario chart.", exc_info=True)

    # Grid search chart (always)
    if len(grid_result.points) > 1:
        try:
            result["grid_search"] = create_grid_search_chart(
                grid_result, charts_dir / "grid_search.png", colors
            )
        except Exception:
            logger.warning("Failed to create grid search chart.", exc_info=True)

    # EEG sensitivity (conditional)
    if eeg_result is not None:
        try:
            result["eeg_sensitivity"] = create_eeg_sensitivity_chart(
                eeg_result, charts_dir / "eeg_sensitivity.png", colors
            )
        except Exception:
            logger.warning("Failed to create EEG sensitivity chart.", exc_info=True)

    # PPA Collar (conditional)
    if collar_result is not None:
        try:
            result["ppa_collar"] = create_ppa_collar_chart(
                collar_result, charts_dir / "ppa_collar.png", colors
            )
        except Exception:
            logger.warning("Failed to create PPA Collar chart.", exc_info=True)

    # PPA Baseload (conditional)
    if baseload_result is not None:
        try:
            result["ppa_baseload"] = create_ppa_baseload_chart(
                baseload_result, charts_dir / "ppa_baseload.png", colors
            )
        except Exception:
            logger.warning("Failed to create PPA Baseload chart.", exc_info=True)

    return result
