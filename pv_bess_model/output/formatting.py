"""Number and currency formatting helpers for output CSVs and stdout.

All functions return strings suitable for writing to CSV files or printing
to the terminal. None values are represented as an empty string.

Public API
----------
fmt_float    – Format a float with configurable decimal places.
fmt_currency – Format a monetary value in euros.
fmt_pct      – Format a fraction as a percentage string.
fmt_optional – Format any optional float, returning "" for None.
"""

from __future__ import annotations

from pv_bess_model.config.defaults import (
    CSV_DECIMAL_SEPARATOR,
    CURRENCY_PRECISION,
    FLOAT_PRECISION,
)


def fmt_float(
    value: float | None,
    precision: int = FLOAT_PRECISION,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> str:
    """Format a float to a fixed number of decimal places.

    Parameters
    ----------
    value:
        The value to format. None is returned as an empty string.
    precision:
        Number of decimal places.
    decimal:
        Decimal separator character (default: ``CSV_DECIMAL_SEPARATOR``).

    Returns
    -------
    str
        Formatted string, e.g. ``"3,1416"`` with the default separator.
    """
    if value is None:
        return ""
    return f"{value:.{precision}f}".replace(".", decimal)


def fmt_currency(
    value: float | None,
    precision: int = CURRENCY_PRECISION,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> str:
    """Format a monetary value in euros.

    Parameters
    ----------
    value:
        Value in euros. None is returned as an empty string.
    precision:
        Decimal places (default 2).
    decimal:
        Decimal separator character (default: ``CSV_DECIMAL_SEPARATOR``).

    Returns
    -------
    str
        Formatted string, e.g. ``"1234567,89"`` with the default separator.
    """
    if value is None:
        return ""
    return f"{value:.{precision}f}".replace(".", decimal)


def fmt_pct(
    value: float | None,
    precision: int = 2,
    *,
    already_pct: bool = False,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> str:
    """Format a fraction (or percentage) as a percentage string.

    Parameters
    ----------
    value:
        The value to format. If ``already_pct=False`` (default), the value is
        treated as a decimal fraction (e.g. 0.0735) and multiplied by 100
        before formatting. If ``already_pct=True``, the value is already in
        percent (e.g. 7.35).
    precision:
        Number of decimal places in the formatted output.
    already_pct:
        Set to True when *value* is already in percent units.
    decimal:
        Decimal separator character (default: ``CSV_DECIMAL_SEPARATOR``).

    Returns
    -------
    str
        Formatted percentage string, e.g. ``"7,35"`` (using default separator,
        without the % sign).
    """
    if value is None:
        return ""
    display = value if already_pct else value * 100.0
    return f"{display:.{precision}f}".replace(".", decimal)


def fmt_optional(
    value: float | None,
    precision: int = FLOAT_PRECISION,
    decimal: str = CSV_DECIMAL_SEPARATOR,
) -> str:
    """Format any optional float, returning an empty string for None.

    Alias for :func:`fmt_float` for cases where the semantic meaning of
    "missing" is more important than the display format.

    Parameters
    ----------
    value:
        Value to format, or None.
    precision:
        Number of decimal places.
    decimal:
        Decimal separator character (default: ``CSV_DECIMAL_SEPARATOR``).

    Returns
    -------
    str
        Formatted string or ``""``.
    """
    return fmt_float(value, precision=precision, decimal=decimal)
