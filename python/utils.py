import time
import multiprocessing as mp

def benchmark(func, *args, **kwargs) -> dict[str, float]:
    '''
    Benchmark a function for performance.

    :param func: The function to benchmark.
    :param *args: Any positional arguments to pass into the function.
    :param **kwargs: Any keyword arguments to pass into the function.
    :return: A dictionary with performance statistics.
    :rtype: dict[str, float]
    '''
    target_sample_size = 1000

    stats = {
        'max': float('-inf'),
        'min': float('inf'),
        'avg': 0,
    }

    for _ in range(target_sample_size):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        stats['max'] = max(stats['max'], elapsed)
        stats['min'] = min(stats['min'], elapsed)
        stats['avg'] += elapsed / target_sample_size

    return stats