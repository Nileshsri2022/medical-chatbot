#!/usr/bin/env python
"""
Test Runner Script for Medical RAG Chatbot
Run tests with various configurations
"""

import sys
import subprocess
import argparse


def run_tests(
    verbose=False,
    markers=None,
    cov=False,
    cov_report="term",
    test_path=None,
    fast_only=False,
):
    """Run pytest with specified options"""

    cmd = ["pytest"]

    if test_path:
        cmd.append(test_path)

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if markers:
        cmd.extend(["-m", markers])

    if fast_only:
        cmd.extend(["-m", "not slow"])

    if cov:
        cmd.extend(
            [
                "--cov=backend",
                f"--cov-report={cov_report}",
                "--cov-fail-under=70",
            ]
        )

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Medical RAG tests")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-m", "--markers", type=str, help="Run tests matching marker")
    parser.add_argument("--cov", action="store_true", help="Run with coverage")
    parser.add_argument(
        "--cov-report",
        type=str,
        default="term",
        choices=["term", "html", "xml"],
        help="Coverage report format",
    )
    parser.add_argument("--test", "-t", type=str, help="Specific test file or path")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--security", action="store_true", help="Run security tests only"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tests with coverage"
    )

    args = parser.parse_args()

    markers = args.markers

    if args.unit:
        markers = "unit"
    elif args.integration:
        markers = "integration"
    elif args.security:
        markers = "security"
    elif args.fast:
        markers = "not slow"

    test_path = args.test or "tests/"

    exit_code = run_tests(
        verbose=args.verbose,
        markers=markers,
        cov=args.cov,
        cov_report=args.cov_report,
        test_path=test_path,
        fast_only=args.fast,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
