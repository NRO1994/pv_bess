"""LLM client for generating report texts via the Anthropic API.

Uses SHA256-based caching to avoid redundant API calls. Falls back to
placeholder text on API errors or missing dependencies.

Public API
----------
LLMClient                    -- API client with caching.
generate_model_description   -- Page 0: model description.
generate_input_summary       -- Page 1: input parameter summary.
generate_pv_yield_text       -- Page 2: PV yield analysis.
generate_price_scenario_text -- Page 3: price scenario analysis.
generate_grid_search_text    -- Page 4: grid search results.
generate_sensitivity_text    -- Pages 5-7: sensitivity analysis.
generate_conclusion          -- Page 8: conclusion.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pv_bess_model.config.defaults import (
    REPORT_LLM_CACHE_FILENAME,
    REPORT_LLM_DEFAULT_MODEL,
    REPORT_LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_FALLBACK_TEXT = "[Textgenerierung fehlgeschlagen]"

_SYSTEM_PROMPT = (
    "Du bist ein sachlicher Finanzanalyst, der professionelle Berichte über "
    "erneuerbare Energieprojekte schreibt. Schreibe auf Deutsch in Fließtext "
    "(3-4 Absätze). Verwende einen nüchternen, faktenbasierten Ton ohne "
    "Übertreibungen. Verwende keine Aufzählungszeichen, sondern formuliere "
    "in ganzen Sätzen. Gib die Analyse direkt wieder, ohne einleitende Floskeln "
    "wie 'Hier ist meine Analyse'."
)


class LLMClient:
    """Anthropic API client with SHA256-based response caching.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    model:
        Model identifier (default: ``REPORT_LLM_DEFAULT_MODEL``).
    cache_dir:
        Directory for the LLM cache JSON file.
    """

    def __init__(
        self,
        api_key: str,
        model: str = REPORT_LLM_DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._cache_dir = cache_dir
        self._cache: dict[str, str] = {}
        self._client: object | None = None

        if cache_dir is not None:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load the cache JSON file if it exists."""
        if self._cache_dir is None:
            return
        cache_path = self._cache_dir / REPORT_LLM_CACHE_FILENAME
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load LLM cache, starting fresh.")
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist the cache to disk."""
        if self._cache_dir is None:
            return
        cache_path = self._cache_dir / REPORT_LLM_CACHE_FILENAME
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save LLM cache.", exc_info=True)

    def _get_client(self) -> object:
        """Lazily import and instantiate the Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = REPORT_LLM_MAX_TOKENS,
    ) -> str:
        """Generate text via the Anthropic API with caching.

        Parameters
        ----------
        system_prompt:
            System-level instruction for the model.
        user_prompt:
            User-level prompt with context data.
        max_tokens:
            Maximum output tokens.

        Returns
        -------
        str
            Generated text, or ``_FALLBACK_TEXT`` on error.
        """
        cache_key = hashlib.sha256(
            (system_prompt + user_prompt + self._model).encode("utf-8")
        ).hexdigest()

        if cache_key in self._cache:
            logger.debug("LLM cache hit for key %s", cache_key[:12])
            return self._cache[cache_key]

        try:
            client = self._get_client()
            response = client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text  # type: ignore[index]
            self._cache[cache_key] = text
            self._save_cache()
            return text
        except Exception:
            logger.warning("LLM API call failed.", exc_info=True)
            return _FALLBACK_TEXT


# ---------------------------------------------------------------------------
# Text generation functions
# ---------------------------------------------------------------------------


def generate_model_description(client: LLMClient, claude_md_excerpt: str) -> str:
    """Generate a model description text for page 0.

    Parameters
    ----------
    client:
        Configured LLM client.
    claude_md_excerpt:
        First ~2000 characters of CLAUDE.md for context.

    Returns
    -------
    str
        Generated model description text.
    """
    prompt = (
        "Beschreibe das folgende Finanzmodell für PV+BESS-Projekte in 3-4 Absätzen. "
        "Erläutere die Methodik (Grid Search, Monte Carlo, LP-Dispatch) und den "
        "Anwendungszweck.\n\n"
        f"Modelldokumentation (Auszug):\n{claude_md_excerpt}"
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_input_summary(client: LLMClient, params: dict) -> str:
    """Generate an input parameter summary text for page 1.

    Parameters
    ----------
    client:
        Configured LLM client.
    params:
        Dictionary of key input parameters.

    Returns
    -------
    str
        Generated summary text.
    """
    params_str = "\n".join(f"- {k}: {v}" for k, v in params.items())
    prompt = (
        "Fasse die folgenden Eingabeparameter eines PV+BESS-Finanzmodells "
        "zusammen und ordne sie ein:\n\n" + params_str
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_pv_yield_text(
    client: LLMClient, annual_kwh_per_year: dict[int, float]
) -> str:
    """Generate PV yield analysis text for page 2.

    Parameters
    ----------
    client:
        Configured LLM client.
    annual_kwh_per_year:
        Mapping ``{weather_year: annual_production_kwh}``.

    Returns
    -------
    str
        Generated analysis text.
    """
    data_str = "\n".join(
        f"- Wetterjahr {y}: {kwh:,.0f} kWh" for y, kwh in sorted(annual_kwh_per_year.items())
    )
    prompt = (
        "Analysiere die folgenden PV-Ertragsdaten aus verschiedenen Wetterjahren. "
        "Gehe auf die Variabilität und Bandbreite ein:\n\n" + data_str
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_price_scenario_text(
    client: LLMClient, scenario_means: dict[str, float]
) -> str:
    """Generate price scenario analysis text for page 3.

    Parameters
    ----------
    client:
        Configured LLM client.
    scenario_means:
        Mapping ``{scenario_name: mean_price_eur_per_mwh}``.

    Returns
    -------
    str
        Generated analysis text.
    """
    data_str = "\n".join(
        f"- {name}: {price:.2f} EUR/MWh" for name, price in sorted(scenario_means.items())
    )
    prompt = (
        "Analysiere die folgenden Strompreis-Szenarien und deren Auswirkungen "
        "auf die Wirtschaftlichkeit eines PV+BESS-Projekts:\n\n" + data_str
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_grid_search_text(
    client: LLMClient,
    optimal_scale: float,
    optimal_ep: float,
    optimal_irr: float,
    pv_only_irr: float | None,
) -> str:
    """Generate grid search results text for page 4.

    Parameters
    ----------
    client:
        Configured LLM client.
    optimal_scale:
        Optimal BESS scale in % of PV.
    optimal_ep:
        Optimal energy-to-power ratio in hours.
    optimal_irr:
        Equity IRR at optimum in %.
    pv_only_irr:
        Equity IRR for PV-only baseline in %, or None.

    Returns
    -------
    str
        Generated analysis text.
    """
    pv_only_str = f"{pv_only_irr:.2f}%" if pv_only_irr is not None else "nicht verfügbar"
    prompt = (
        f"Analysiere die Ergebnisse der BESS-Dimensionierungsoptimierung:\n\n"
        f"- Optimale BESS-Skalierung: {optimal_scale:.0f}% der PV-Leistung\n"
        f"- Optimales E/P-Verhältnis: {optimal_ep:.1f} Stunden\n"
        f"- Equity IRR am Optimum: {optimal_irr:.2f}%\n"
        f"- Equity IRR ohne BESS (PV-Only): {pv_only_str}\n\n"
        f"Bewerte die Vorteilhaftigkeit der BESS-Integration."
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_sensitivity_text(
    client: LLMClient, analysis_type: str, key_findings: str
) -> str:
    """Generate sensitivity analysis text for pages 5-7.

    Parameters
    ----------
    client:
        Configured LLM client.
    analysis_type:
        Type of analysis (e.g. "EEG-Sensitivität").
    key_findings:
        Pre-formatted key findings string.

    Returns
    -------
    str
        Generated analysis text.
    """
    prompt = (
        f"Analysiere die Ergebnisse der folgenden Sensitivitätsanalyse:\n\n"
        f"Analysetyp: {analysis_type}\n\n"
        f"Ergebnisse:\n{key_findings}"
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt)


def generate_conclusion(
    client: LLMClient, all_results_summary: str
) -> str:
    """Generate a conclusion text for page 8.

    Parameters
    ----------
    client:
        Configured LLM client.
    all_results_summary:
        Pre-formatted summary of all results.

    Returns
    -------
    str
        Generated conclusion text (up to 700 tokens).
    """
    prompt = (
        "Schreibe eine abschließende Bewertung und Handlungsempfehlung für das "
        "folgende PV+BESS-Projekt basierend auf den Analyseergebnissen:\n\n"
        + all_results_summary
    )
    return client.generate_text(_SYSTEM_PROMPT, prompt, max_tokens=700)
