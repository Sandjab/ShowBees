import os
import random
from datetime import datetime

import numpy as np
import psutil
import tensorflow as tf
from pyprojroot import here

rootpath = here(project_files=['.kilroy'], warn=False)


def mooltipath(*args):
    return rootpath.joinpath(*args)


def predestination(seed_value=23081965):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    # tf.random.set_seed(seed_value)
    # for later versions:
    tf.compat.v1.set_random_seed(seed_value)


def iprint(*args, **kwargs):
    t = datetime.now().strftime('%Y-%m-%d/%H:%M:%S.%f')[:-3]
    ucpu = psutil.cpu_percent(interval=None, percpu=False)
    uram = psutil.virtual_memory().percent
    process_uram = psutil.Process().memory_info().rss/1024/1024/1024
    print('[{0}|{1:04.1f}%|{2:04.1f}%|{3:.2f}GB]'.format(
        t, ucpu, uram, process_uram), *args, **kwargs)
