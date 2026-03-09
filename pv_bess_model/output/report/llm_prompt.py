"""LLM prompt rendering and response parsing for the HTML report.

Replaces the Anthropic API-based text generation with a manual Copilot
workflow: the prompt is rendered to a file, the user pastes it into
Copilot, and saves the JSON response which is then loaded back.

Public API
----------
render_prompt           -- Fill the prompt template with report data.
save_rendered_prompt    -- Write the filled prompt to the output directory.
load_llm_response       -- Load and validate a JSON response file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pv_bess_model.config.defaults import (
    REPORT_LLM_PROMPT_FILENAME,
    REPORT_LLM_RESPONSE_FILENAME,
)

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent.parent.parent.parent / ".docs" / "llm_templates" / "report_prompt.md"

_EXPECTED_KEYS = [
    "tab_1_overview",
    "tab_2_timeseries",
    "tab_3_gridsearch",
    "tab_4_eeg",
    "tab_5_collar",
    "tab_6_baseload",
    "tab_7_cashflow",
]

_FALLBACK_TEXT = (
    "CoPilot nicht verfügbar. "
    "Bitte führen Sie den LLM-Prompt-Workflow manuell durch."
)


def render_prompt(data: Any) -> str:
    """Fill the prompt template with values from ``HtmlReportData``.

    Parameters
    ----------
    data:
        An ``HtmlReportData`` instance.

    Returns
    -------
    str
        The fully rendered prompt string.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    # Marketing details
    marketing_lines: list[str] = []
    mp = data.marketing_params
    if "floor_price_ct_kwh" in mp:
        marketing_lines.append(f"- Floor-Preis: {mp['floor_price_ct_kwh']:.2f} ct/kWh")
    if "cap_price_ct_kwh" in mp:
        marketing_lines.append(f"- Cap-Preis: {mp['cap_price_ct_kwh']:.2f} ct/kWh")
    if "fixed_price_years" in mp:
        marketing_lines.append(f"- Förderdauer: {mp['fixed_price_years']} Jahre")
    if "ppa_duration_years" in mp:
        marketing_lines.append(f"- PPA-Laufzeit: {mp['ppa_duration_years']} Jahre")
    if "goo_premium_ct_kwh" in mp:
        marketing_lines.append(f"- GoO-Praemie: {mp['goo_premium_ct_kwh']:.2f} ct/kWh")
    if "baseload_mw" in mp:
        marketing_lines.append(f"- Baseload: {mp['baseload_mw']} MW")
    if "ppa_price_ct_kwh" in mp:
        marketing_lines.append(f"- PPA-Preis: {mp['ppa_price_ct_kwh']:.2f} ct/kWh")

    # Weather years
    weather_years: list[str] = []
    for year, prod in data.pv_monthly_by_year.items():
        winter_mean_prod = (sum(prod[:3]) + sum(prod[9:])) / 6
        summer_mean_prod = sum(prod[3:9]) / 6
        weather_years.append(f"  - {year}: durchschnittliche Monatsproduktion im Winter: {winter_mean_prod:.2f} GWh,"
                             f" im Sommer {summer_mean_prod:.2f} GWh")
    weather_stats = "\n".join(weather_years) if weather_years else ""

    # Price scenarios summary
    price_summaries: list[str] = []
    for ps in data.price_scenario_annual_means:
        if ps["means"]:
            avg_y5 = sum(ps["means"][:5]) / 5
            avg_y10 = sum(ps["means"][10:15]) / 5
            avg_tail = sum(ps["means"][-5:]) / 5
            price_summaries.append(f"  - {ps['name']}: Mittlerer Spotpreis "
                                   f"Jahr {data.commissioning_year} - {data.commissioning_year + 5}: {avg_y5:.2f} EUR/MWh, "
                                   f"Jahr {data.commissioning_year+ 10} - {data.commissioning_year + 15}: {avg_y10:.2f} EUR/MWh, "
                                   f"Jahr {data.commissioning_year + data.lifetime_years -5} - {data.commissioning_year + data.lifetime_years}: {avg_tail:.2f} EUR/MWh")
    price_scenarios_summary = "\n".join(price_summaries) if price_summaries else "keine"

    # Metrics formatting
    m = data.metrics
    equity_irr = f"{m['equity_irr']:.2f} %" if m.get("equity_irr") is not None else "n/a"
    project_irr = f"{m['project_irr']:.2f} %" if m.get("project_irr") is not None else "n/a"
    npv = f"{m['npv']:,.0f} EUR" if m.get("npv") is not None else "n/a"
    dscr_min = f"{m['dscr_min']:.2f}" if m.get("dscr_min") is not None else "n/a"
    dscr_avg = f"{m['dscr_avg']:.2f}" if m.get("dscr_avg") is not None else "n/a"
    lcoe = f"{m['lcoe']:.3f} ct/kWh" if m.get("lcoe") is not None else "n/a"
    payback = str(m["payback_year"]) if m.get("payback_year") is not None else "n/a"

    # Sensitivity section
    sens_lines: list[str] = []
    if data.eeg_sensitivity:
        sens_lines.append("### EEG-Sensitivitaet (20 Jahre Laufzeit)")
        for pt in data.eeg_sensitivity:
            floor = pt.get("floor_price_eur_per_kwh", 0) * 100
            irr_mean = pt.get("irr_mean", 0)
            irr_std = pt.get("equity_irr_std", 0)
            sens_lines.append(f"- Floor {floor:.2f} ct/kWh -> "
                              f"durchschnittlicher IRR {irr_mean:.2f} %, Std.Abweichung eq.IRR {irr_std:.2f} %")
    if data.ppa_collar:
        sens_lines.append(f"### PPA-Collar-Analyse ({data.ppa_collar_duration} Jahre Laufzeit)")
        for pt in data.ppa_collar:
            floor = pt.get("floor_price_eur_per_kwh", 0) * 100
            cap = pt.get("cap_price_eur_per_kwh", 0) * 100
            irr_mean = pt.get("irr_mean", 0)
            irr_std = pt.get("irr_std", 0)
            sens_lines.append(f"- Floor {floor:.2f} ct/kWh, Cap {cap:.2f} ct/kWh -> "
                              f"durchschnittlicher eq.IRR {irr_mean:.2f} %, Std.Abweichung eq.IRR {irr_std:.2f} %")
    if data.ppa_baseload:
        sens_lines.append(f"### PPA-Baseload-Analyse ({data.ppa_baseload_duration} Jahre Laufzeit)")
        for pt in data.ppa_baseload:
            irr_mean = pt.get("irr_mean", 0)
            irr_std = pt.get("irr_std", 0)
            baseload = pt.get("baseload_mw", 0)
            ppa_price = pt.get("ppa_price_eur_per_kwh", 0) * 100
            sens_lines.append(f"- Baseload {baseload} MW, PPA-Preis {ppa_price:.2f} ct/kWh -> "
                              f"durchschnittlicher IRR {irr_mean:.2f} %, Std.Abweichung eq.IRR {irr_std:.2f} %")

    sensitivity_section = "\n".join(sens_lines) if sens_lines else ""

    # Substitutions
    replacements = {
        "{{scenario_name}}": data.scenario_name,
        "{{creation_date}}": data.creation_date,
        "{{commissioning_year}}": str(data.commissioning_year),
        "{{scenario_json_filename}}": data.scenario_json_filename,
        "{{pv_peak_kwp}}": f"{data.pv_peak_kwp:,.0f}",
        "{{pv_azimuth}}": f"{data.pv_azimuth:.0f}",
        "{{pv_tilt}}": f"{data.pv_tilt:.0f}",
        "{{pv_degradation_pct}}": f"{data.pv_degradation_pct:.1f}",
        "{{bess_rte_pct}}": f"{data.bess_rte_pct:.0f}",
        "{{grid_max_export_kw}}": f"{data.grid_max_export_kw:,.0f}",
        "{{operating_mode}}": data.operating_mode,
        "{{latitude}}": f"{data.latitude:.4f}",
        "{{longitude}}": f"{data.longitude:.4f}",
        "{{lifetime_years}}": str(data.lifetime_years),
        "{{leverage_pct}}": f"{data.leverage_pct:.0f}",
        "{{interest_rate_pct}}": f"{data.interest_rate_pct:.1f}",
        "{{loan_tenor_years}}": str(data.loan_tenor_years),
        "{{inflation_rate_pct}}": f"{data.inflation_rate * 100:.1f}",
        "{{marketing_type}}": data.marketing_type,
        "{{marketing_details}}": "\n".join(marketing_lines),
        "{{optimal_scale_pct}}": f"{data.optimal_scale_pct:.0f}",
        "{{optimal_ep_ratio}}": f"{data.optimal_ep_ratio:.1f}",
        "{{optimal_bess_power_kw}}": f"{data.optimal_bess_power_kw:,.0f}",
        "{{optimal_bess_capacity_kwh}}": f"{data.optimal_bess_capacity_kwh:,.0f}",
        "{{grid_search_count}}": str(len(data.grid_search_points)),
        "{{equity_irr}}": equity_irr,
        "{{project_irr}}": project_irr,
        "{{npv}}": npv,
        "{{dscr_min}}": dscr_min,
        "{{dscr_avg}}": dscr_avg,
        "{{lcoe}}": lcoe,
        "{{payback_year}}": payback,
        "{{pv_production_model}}": data.pv_production_model,
        "{{price_origin}}": data.price_origin,
        "{{weather_years}}": weather_stats,
        "{{price_scenarios_summary}}": price_scenarios_summary,
        "{{sensitivity_section}}": sensitivity_section,
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


def save_rendered_prompt(
        data: Any,
        output_dir: Path,
) -> Path:
    """Render the LLM prompt and save it to the output directory.

    Parameters
    ----------
    data:
        An ``HtmlReportData`` instance.
    output_dir:
        Output directory for the prompt file.

    Returns
    -------
    Path
        Path to the saved prompt file.
    """
    prompt_text = render_prompt(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"{data.scenario_name}{REPORT_LLM_PROMPT_FILENAME}"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    logger.info("LLM prompt saved to %s", prompt_path)
    return prompt_path


def load_llm_response(
        response_path: Path,
) -> dict[str, str | None]:
    """Load and validate an LLM response JSON file.

    Parameters
    ----------
    response_path:
        Path to the JSON file containing LLM-generated texts.

    Returns
    -------
    dict[str, str | None]
        Mapping of tab keys to text content. Missing keys are filled
        with ``_FALLBACK_TEXT``.

    Raises
    ------
    ValueError
        If the file does not contain valid JSON or is not a dict.
    """
    raw = response_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM-Antwort ist kein gueltiges JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "LLM-Antwort muss ein JSON-Objekt sein, kein "
            f"{type(parsed).__name__}."
        )

    # Validate and fill missing keys
    result: dict[str, str | None] = {}
    for key in _EXPECTED_KEYS:
        if key not in parsed:
            # Key entirely missing → use fallback text
            logger.warning("LLM response missing key '%s', using fallback.", key)
            result[key] = _FALLBACK_TEXT
        else:
            value = parsed[key]
            if value is None:
                result[key] = None  # Tab explicitly null (not applicable)
            elif isinstance(value, str):
                result[key] = value
            else:
                logger.warning("LLM response key '%s' is not a string, using fallback.", key)
                result[key] = _FALLBACK_TEXT

    return result


def get_fallback_texts() -> dict[str, str]:
    """Return fallback placeholder texts for all tabs.

    Returns
    -------
    dict[str, str]
        Mapping of tab keys to fallback text.
    """
    return {key: _FALLBACK_TEXT for key in _EXPECTED_KEYS}
