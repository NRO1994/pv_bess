"""BDEW standard load profile (SLP) generation and caching.

Generates 35,040 quarter-hourly load profile values from BDEW coefficients,
applying day-type assignment (Werktag/Samstag/Sonn- und Feiertag) and
dynamization functions for supported profile types (H25, P25, S25).

Implementation planned for Phase 2.
"""
