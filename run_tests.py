"""
Script to run all unit tests.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_momentum import run_all_momentum_tests
from tests.test_performance import run_all_performance_tests
from tests.test_statistical_tests import run_all_statistical_tests


def main():
    """
    Run all unit tests.
    """
    print("\n" + "=" * 80)
    print("RUNNING ALL UNIT TESTS")
    print("=" * 80)
    print()
    
    success = True
    
    try:
        run_all_momentum_tests()
    except Exception as e:
        print(f"\n✗ Momentum tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        run_all_performance_tests()
    except Exception as e:
        print(f"\n✗ Performance tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        run_all_statistical_tests()
    except Exception as e:
        print(f"\n✗ Statistical tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
