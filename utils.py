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
    # The total runtime of the benchmark should not exceed three minutes
    total_max_sec = 180
    total_elapsed_time = 0
    sample_size = 0
    target_sample_size = 1000

    stats = {
        'max': float('-inf'),
        'min': float('inf'),
        'avg': 0,
    }

    for i in range(target_sample_size):
        # Check to see how much time we have
        remaining_time = total_max_sec - total_elapsed_time
        if remaining_time <= 0:
            break

        # Measure the runtime of a single instance of the function
        p = mp.Process(target=func, args=args, kwargs=kwargs)
        start = time.time()
        p.start()
        p.join(timeout=remaining_time)
        end = time.time()

        # If the timeout was exceeded, print an error message and break from the calculation
        if p.is_alive():
            print(f'Function timed out after {remaining_time} seconds on iteration {i+1}')
            p.terminate()
            p.join()
            break

        # Calculate elapsed time and update the min/max
        elapsed = end - start
        stats['max'] = max(stats['max'], elapsed)
        stats['min'] = min(stats['min'], elapsed)

        # Update sample size and total elapsed time for average calculation
        sample_size += 1
        total_elapsed_time += elapsed
    
    # Calculate average runtime
    if sample_size > 0:
        stats['avg'] = total_elapsed_time / sample_size

    return stats