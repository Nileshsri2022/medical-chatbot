"""
Performance Profiling Module
Utilities for profiling and optimizing RAG engine performance
"""

import time
import functools
import cProfile
import pstats
import io
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class ProfileResult:
    """Result of profiling an operation"""

    function_name: str
    total_time: float
    call_count: int
    stats: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Performance profiler for RAG engine operations"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.results: Dict[str, ProfileResult] = {}

    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling a code block"""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        profiler = cProfile.Profile()

        try:
            profiler.enable()
            yield
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - start_time

            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            sort_stats = stats.sort_stats("cumulative")

            self.results[operation_name] = ProfileResult(
                function_name=operation_name,
                total_time=elapsed,
                call_count=stats.total_calls,
                stats=self._extract_stats(sort_stats),
            )

    def _extract_stats(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Extract useful stats from profiler"""
        return {
            "top_functions": [
                {"function": f.function, "cumulative_time": f.cumtime}
                for f in list(stats.fcn_list)[:5]
            ]
            if hasattr(stats, "fcn_list")
            else []
        }

    def get_results(self) -> Dict[str, ProfileResult]:
        return self.results

    def print_results(self):
        """Print profiling results"""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"\n📊 {name}")
            print(f"   Time: {result.total_time:.4f}s")
            print(f"   Calls: {result.call_count}")

        print("\n" + "=" * 60)

    def reset(self):
        """Reset profiling results"""
        self.results = {}


_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    return _profiler


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        print(f"⏱️ {func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper


def timeit(n_iterations: int = 10):
    """Decorator to time a function over multiple iterations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"⏱️ {func.__name__} ({n_iterations} iterations)")
            print(
                f"   Avg: {avg_time:.4f}s | Min: {min_time:.4f}s | Max: {max_time:.4f}s"
            )
            return result

        return wrapper

    return decorator


class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"⏱️ {self.name}: {self.elapsed:.4f}s")
        return False


def benchmark(
    cases: Dict[str, Callable], n_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """Benchmark multiple functions"""
    results = {}

    for name, func in cases.items():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)

        results[name] = {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times),
        }

    return results


def print_benchmark_results(results: Dict[str, Dict[str, float]]):
    """Print benchmark results in a nice format"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for name, stats in results.items():
        print(f"\n📊 {name}")
        print(f"   Avg: {stats['avg']:.6f}s")
        print(f"   Min: {stats['min']:.6f}s")
        print(f"   Max: {stats['max']:.6f}s")

    print("\n" + "=" * 60)
