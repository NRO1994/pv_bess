"""HTML report builder – assembles the interactive single-file dashboard.

Loads the ``dashboard.html`` template, injects the serialised
``HtmlReportData`` as JSON, and writes the final HTML file.

Public API
----------
build_html_report            -- Create the interactive HTML report (PV+BESS).
build_portfolio_html_report  -- Create the portfolio/Systemwert HTML report.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pv_bess_model.config.defaults import REPORT_HTML_FILENAME_SUFFIX

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def build_html_report(
    data: object,
    output_dir: Path,
) -> Path:
    """Create the interactive HTML report.

    1. Load the template from ``templates/dashboard.html``.
    2. Serialise ``HtmlReportData`` as compact JSON (NaN/Inf → null).
    3. Replace ``{{REPORT_DATA_JSON}}`` in the template.
    4. Replace ``{{scenario_name}}`` in the ``<title>``.
    5. Write the finished HTML file to *output_dir*.

    Parameters
    ----------
    data:
        An ``HtmlReportData`` instance (from ``data_collector``).
    output_dir:
        Target directory for the report file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    template_path = _TEMPLATE_DIR / "dashboard.html"
    template = template_path.read_text(encoding="utf-8")

    data_json = data.to_json()  # type: ignore[union-attr]

    html = template.replace("{{REPORT_DATA_JSON}}", data_json)
    html = html.replace("{{scenario_name}}", data.scenario_name)  # type: ignore[union-attr]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{data.scenario_name}{REPORT_HTML_FILENAME_SUFFIX}"  # type: ignore[union-attr]
    report_path.write_text(html, encoding="utf-8")

    logger.info("HTML report written to: %s", report_path)
    return report_path


def build_portfolio_html_report(
    data: object,
    output_dir: Path,
) -> Path:
    """Create the portfolio/Systemwert interactive HTML report.

    1. Load the template from ``templates/dashboard_portfolio.html``.
    2. Serialise ``PortfolioReportData`` as compact JSON (NaN/Inf → null).
    3. Replace ``{{REPORT_DATA_JSON}}`` in the template.
    4. Replace ``{{scenario_name}}`` in the ``<title>``.
    5. Write the finished HTML file to *output_dir*.

    Parameters
    ----------
    data:
        A ``PortfolioReportData`` instance (from ``data_collector_portfolio``).
    output_dir:
        Target directory for the report file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    template_path = _TEMPLATE_DIR / "dashboard_portfolio.html"
    template = template_path.read_text(encoding="utf-8")

    data_json = data.to_json()  # type: ignore[union-attr]

    html = template.replace("{{REPORT_DATA_JSON}}", data_json)
    html = html.replace("{{scenario_name}}", data.scenario_name)  # type: ignore[union-attr]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{data.scenario_name}_portfolio_report.html"  # type: ignore[union-attr]
    report_path.write_text(html, encoding="utf-8")

    logger.info("Portfolio HTML report written to: %s", report_path)
    return report_path
