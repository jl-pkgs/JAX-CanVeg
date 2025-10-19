# how to run all tests
def run_tests():
    import pytest
    import sys

    exit_code = pytest.main(sys.argv[1:] if len(sys.argv) > 1 else [])
    sys.exit(exit_code)


if __name__ == "__main__":
    run_tests()
