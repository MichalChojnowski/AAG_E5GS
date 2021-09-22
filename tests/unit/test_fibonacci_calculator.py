import math
import pytest
from recurrence_calculators import FibonacciCalculator


# Helper function local to these tests computes approximate fibonacci formula
def fibonacci_formula(n):
    phi = (1 + math.sqrt(5)) / 2
    neg = (1 - math.sqrt(5)) / 2
    return int((math.pow(phi, n + 1) - math.pow(neg, n + 1)) / math.sqrt(5))


# This is an example of parametrization, this is 4 tests one for each value
@pytest.mark.parametrize("n", [5, 8, 10, 33])
def test_fibonacci_calculator(n):
    assert FibonacciCalculator().compute(n) == fibonacci_formula(n)


# This is an example of a mark or name of a group of tests
# You can run only these two tests with the mark 'cache_tests' using the -k option
# `pytest -k cache_tests` or the inverse `pytest -k "not cache_tests"`
@pytest.mark.cache_tests
def test_cache_count():
    fc = FibonacciCalculator()
    fc.compute(7)
    assert fc.count() == 8


@pytest.mark.cache_tests
def test_new_cache_count():
    fc = FibonacciCalculator()
    assert fc.count() == 2  # from the initial 2 starting elements in sequence


@pytest.mark.seed_tests
def test_reset_fibonacci_seed():
    fc = FibonacciCalculator()
    fc.reset_start([1, 3, 4, 7, 13, 20])
    assert fc.count() == 6
    assert fc.compute(10) == 225
    assert fc.count() == 11
