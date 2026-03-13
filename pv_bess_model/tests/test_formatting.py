"""Tests for output/formatting.py – fmt_float, fmt_currency, fmt_pct, fmt_optional."""

from __future__ import annotations

import pytest

from pv_bess_model.output.formatting import (
    fmt_currency,
    fmt_float,
    fmt_optional,
    fmt_pct,
)


class TestFmtFloat:
    def test_basic_value(self) -> None:
        result = fmt_float(3.14159, precision=2, decimal=".")
        assert result == "3.14"

    def test_none_returns_empty(self) -> None:
        assert fmt_float(None) == ""

    def test_comma_decimal(self) -> None:
        result = fmt_float(1.5, precision=1, decimal=",")
        assert result == "1,5"

    def test_zero(self) -> None:
        result = fmt_float(0.0, precision=2, decimal=".")
        assert result == "0.00"

    def test_negative(self) -> None:
        result = fmt_float(-42.123, precision=2, decimal=".")
        assert result == "-42.12"


class TestFmtCurrency:
    def test_basic_value(self) -> None:
        result = fmt_currency(1234567.89, precision=2, decimal=".")
        assert result == "1234567.89"

    def test_none_returns_empty(self) -> None:
        assert fmt_currency(None) == ""

    def test_comma_decimal(self) -> None:
        result = fmt_currency(99.5, precision=2, decimal=",")
        assert result == "99,50"

    def test_zero(self) -> None:
        result = fmt_currency(0.0, decimal=".")
        assert result == "0.00"


class TestFmtPct:
    def test_fraction_to_pct(self) -> None:
        result = fmt_pct(0.0735, precision=2, decimal=".")
        assert result == "7.35"

    def test_already_pct(self) -> None:
        result = fmt_pct(7.35, precision=2, already_pct=True, decimal=".")
        assert result == "7.35"

    def test_none_returns_empty(self) -> None:
        assert fmt_pct(None) == ""

    def test_zero(self) -> None:
        result = fmt_pct(0.0, precision=2, decimal=".")
        assert result == "0.00"

    def test_comma_decimal(self) -> None:
        result = fmt_pct(0.1234, precision=2, decimal=",")
        assert result == "12,34"

    def test_negative_fraction(self) -> None:
        result = fmt_pct(-0.05, precision=2, decimal=".")
        assert result == "-5.00"


class TestFmtOptional:
    def test_basic_value(self) -> None:
        result = fmt_optional(3.14, precision=2, decimal=".")
        assert result == "3.14"

    def test_none_returns_empty(self) -> None:
        assert fmt_optional(None) == ""

    def test_delegates_to_fmt_float(self) -> None:
        assert fmt_optional(1.5, precision=1, decimal=",") == fmt_float(1.5, precision=1, decimal=",")
