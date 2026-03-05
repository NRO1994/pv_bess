"""PDF report assembly from HTML template, charts, and LLM texts.

Loads the HTML template, substitutes placeholders, and renders to PDF
using weasyprint.

Public API
----------
ReportConfig  -- Configuration dataclass for report generation.
build_report  -- Assemble and render the PDF report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pv_bess_model.config.defaults import (
    REPORT_MODEL_VERSION,
    REPORT_PDF_FILENAME_SUFFIX,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_NO_LLM_TEXT = "[LLM-Text nicht verfügbar]"


@dataclass
class ReportConfig:
    """Configuration for PDF report generation.

    Attributes
    ----------
    enabled:
        Whether report generation is enabled.
    company_name:
        Company name displayed on the cover page.
    logo_path:
        Optional path to a company logo image file.
    """

    enabled: bool = False
    company_name: str = ""
    logo_path: str | None = None


def _build_params_table(scenario: Any) -> str:
    """Build an HTML table of key scenario parameters.

    Parameters
    ----------
    scenario:
        ``ScenarioConfig`` instance.

    Returns
    -------
    str
        HTML table string.
    """
    pv = scenario.pv
    pv_design = pv.get("design", {})
    pv_perf = pv.get("performance", {})
    bess = scenario.bess
    bess_perf = bess.get("performance", {})
    bess_design = bess.get("design_space", {})
    grid = scenario.grid_connection
    finance = scenario.finance

    rows: list[tuple[str, str, str]] = []

    # PV section
    rows.append(("section", "Photovoltaik", ""))
    rows.append(("", "Nennleistung", f"{pv_design.get('peak_power_kwp', 0):,.0f} kWp"))
    rows.append(("", "Ausrichtung", f"{pv_design.get('azimuth_deg', 0)}° / {pv_design.get('tilt_deg', 0)}°"))
    rows.append(("", "Degradation", f"{pv_perf.get('degradation_rate_pct_per_year', 0.4)}%/a"))

    # BESS section
    rows.append(("section", "Batteriespeicher", ""))
    rows.append(("", "Skalierung (% PV)", ", ".join(str(s) for s in bess_design.get("scale_pct_of_pv", []))))
    rows.append(("", "E/P-Verhältnis (h)", ", ".join(str(e) for e in bess_design.get("e_to_p_ratio_hours", []))))
    rows.append(("", "Round-Trip-Effizienz", f"{bess_perf.get('round_trip_efficiency_pct', 88)}%"))
    rows.append(("", "Degradation", f"{bess_perf.get('degradation_rate_pct_per_year', 2)}%/a"))
    rows.append(("", "Verfügbarkeit", f"{bess_perf.get('bess_availability_pct', 97)}%"))

    # Grid section
    rows.append(("section", "Netzanschluss", ""))
    rows.append(("", "Max. Einspeiseleistung", f"{grid.get('max_export_kw', 0):,.0f} kW"))

    # Finance section
    rows.append(("section", "Finanzierung", ""))
    rows.append(("", "Projektlaufzeit", f"{scenario.lifetime_years} Jahre"))
    rows.append(("", "Inbetriebnahme", str(scenario.commissioning_year)))
    rows.append(("", "Fremdkapitalquote", f"{finance.get('leverage_pct', 0)}%"))
    rows.append(("", "Zinssatz", f"{finance.get('interest_rate_pct', 0)}%"))
    rows.append(("", "Kreditlaufzeit", f"{finance.get('loan_tenor_years', 0)} Jahre"))
    rows.append(("", "Inflationsrate", f"{finance.get('inflation_rate', 0) * 100:.1f}%"))

    # Marketing
    marketing = finance.get("revenue_streams", {}).get("marketing", {})
    rows.append(("section", "Vermarktung", ""))
    rows.append(("", "Vermarktungsmodell", marketing.get("type", "market")))
    floor = marketing.get("floor_price_eur_per_kwh")
    if floor is not None:
        rows.append(("", "Mindestpreis", f"{floor * 100:.2f} ct/kWh"))
    fixed_years = marketing.get("fixed_price_years")
    if fixed_years is not None:
        rows.append(("", "Festpreislaufzeit", f"{fixed_years} Jahre"))

    # Build HTML
    html_parts = ['<table class="params-table">']
    html_parts.append("<tr><th>Parameter</th><th>Wert</th></tr>")
    for row_type, label, value in rows:
        if row_type == "section":
            html_parts.append(
                f'<tr class="section-header"><td colspan="2">{label}</td></tr>'
            )
        else:
            html_parts.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    html_parts.append("</table>")
    return "\n".join(html_parts)


def _build_conditional_page(title: str, chart_path: Path | None, text: str) -> str:
    """Build an optional page section. Returns empty string if no chart.

    Parameters
    ----------
    title:
        Page heading.
    chart_path:
        Path to the chart PNG, or None to skip.
    text:
        Descriptive text for the page.

    Returns
    -------
    str
        HTML section string, or empty string if chart_path is None.
    """
    if chart_path is None:
        return ""

    return (
        f'<section class="page">\n'
        f"    <h2>{title}</h2>\n"
        f'    <div class="chart-container">\n'
        f'        <img src="{chart_path.resolve()}" alt="{title}">\n'
        f"    </div>\n"
        f'    <div class="text-content">\n'
        f"        {text}\n"
        f"    </div>\n"
        f"</section>\n"
    )


def build_report(
    scenario_name: str,
    output_dir: Path,
    chart_paths: dict[str, Path],
    texts: dict[str, str],
    config: ReportConfig,
    scenario: Any,
) -> Path | None:
    """Assemble and render the PDF report.

    Parameters
    ----------
    scenario_name:
        Name of the scenario for the filename.
    output_dir:
        Directory where the PDF will be saved.
    chart_paths:
        Mapping of chart name to PNG file path (from ``create_all_charts``).
    texts:
        Mapping of text placeholder name to generated text.
    config:
        Report configuration.
    scenario:
        ``ScenarioConfig`` instance for parameter table.

    Returns
    -------
    Path or None
        Path to the generated PDF, or None if rendering failed.
    """
    template_path = _TEMPLATE_DIR / "report.html"
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError:
        logger.error("Report template not found: %s", template_path)
        return None

    # Logo
    logo_html = ""
    if config.logo_path:
        logo_file = Path(config.logo_path)
        if logo_file.exists():
            logo_html = f'<img class="logo" src="{logo_file.resolve()}" alt="Logo">'
        else:
            logger.warning("Logo file not found: %s", config.logo_path)

    # Parameter table
    params_table_html = _build_params_table(scenario)

    # Grid search chart (always present)
    grid_search_chart = chart_paths.get("grid_search")
    chart_grid_search_src = str(grid_search_chart.resolve()) if grid_search_chart else ""

    # Conditional pages
    pv_yield_page = _build_conditional_page(
        "PV-Ertrag nach Wetterjahr",
        chart_paths.get("pv_yield"),
        texts.get("text_pv_yield", _NO_LLM_TEXT),
    )
    price_scenario_page = _build_conditional_page(
        "Preisszenarien",
        chart_paths.get("price_scenarios"),
        texts.get("text_price_scenarios", _NO_LLM_TEXT),
    )
    eeg_sensitivity_page = _build_conditional_page(
        "EEG-Sensitivitätsanalyse",
        chart_paths.get("eeg_sensitivity"),
        texts.get("text_eeg_sensitivity", _NO_LLM_TEXT),
    )
    ppa_collar_page = _build_conditional_page(
        "PPA Collar-Analyse",
        chart_paths.get("ppa_collar"),
        texts.get("text_ppa_collar", _NO_LLM_TEXT),
    )
    ppa_baseload_page = _build_conditional_page(
        "PPA Baseload-Analyse",
        chart_paths.get("ppa_baseload"),
        texts.get("text_ppa_baseload", _NO_LLM_TEXT),
    )

    # Substitute placeholders
    replacements = {
        "{logo_html}": logo_html,
        "{project_name}": scenario_name,
        "{company_name}": config.company_name or "",
        "{report_date}": date.today().strftime("%d.%m.%Y"),
        "{model_version}": REPORT_MODEL_VERSION,
        "{text_model_description}": texts.get("text_model_description", _NO_LLM_TEXT),
        "{params_table_html}": params_table_html,
        "{text_input_summary}": texts.get("text_input_summary", _NO_LLM_TEXT),
        "{pv_yield_page}": pv_yield_page,
        "{price_scenario_page}": price_scenario_page,
        "{chart_grid_search}": chart_grid_search_src,
        "{text_grid_search}": texts.get("text_grid_search", _NO_LLM_TEXT),
        "{eeg_sensitivity_page}": eeg_sensitivity_page,
        "{ppa_collar_page}": ppa_collar_page,
        "{ppa_baseload_page}": ppa_baseload_page,
        "{text_conclusion}": texts.get("text_conclusion", _NO_LLM_TEXT),
    }

    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # Render PDF
    pdf_path = output_dir / f"{scenario_name}{REPORT_PDF_FILENAME_SUFFIX}"
    try:
        import weasyprint

        doc = weasyprint.HTML(string=html, base_url=str(output_dir.resolve()))
        doc.write_pdf(str(pdf_path))
        logger.info("PDF report written to: %s", pdf_path)
        return pdf_path
    except ImportError:
        logger.warning(
            "weasyprint not installed. Skipping PDF rendering. "
            "Charts have been saved as PNGs."
        )
        return None
    except Exception:
        logger.error("PDF rendering failed.", exc_info=True)
        return None
