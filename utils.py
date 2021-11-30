import random
import time
import functools
import logging
import numpy as np
import torch
import colorama
from itertools import tee


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def bootstrap(data, func=np.mean, n_boot=10000, seed=None):
    n = len(data)
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    integers = rng.integers
    
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [data.take(resampler, axis=0)]
        boot_dist.append(func(*sample))
        
    return np.array(boot_dist)


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        logging.info(f'Total time spent in {str(func.__name__)}: {(end - start):.2f} seconds.')
        return out

    return wrapper


def colored_text(msg, color, style='normal'):
    color = colorama.Fore.__dict__[color.upper()]
    style = colorama.Style.__dict__[style.upper()]
    text = style + color + msg + colorama.Style.RESET_ALL
    return text
