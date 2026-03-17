"""Utility functions for Lecture 1. Available in the notebook via `from utils import *`."""


def check(name, got, expected):
    if got == expected:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} - expected {expected!r}, got {got!r}")
