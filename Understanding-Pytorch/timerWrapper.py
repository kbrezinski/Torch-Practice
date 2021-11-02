
import functools
import time

def Timer(log):
    def wrapper(f):
        @functools.wraps(f)
        def logging_f(*args, **kwargs):
            start = time.perf_counter()
            f(*args, **kwargs)
            total = time.perf_counter() - start
            if log:
                logger.append(total)
        return logging_f
    return wrapper

logger = []

@Timer(log=True)
def test(n: int = 1e5):
    while n < 0:
        n -= 1

test()

print(logger)
