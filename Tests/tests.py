#tests
import cv2
import numpy as np
from FL_lib.find_lines import find_lines
from Tests.test1 import run_test_1
from Tests.test2 import run_test_2


def run_tests(test_params):
    print("Running tests...")
    passed = 0
    total = 0

    if run_test_1(test_params): passed += 1; total += 1
    if run_test_2(test_params): passed += 1; total += 1

    if passed == total:
        print("All tests passed!")
    else:
        print(f"{passed}/{total} tests passed.")
    return passed, total
