import random
from time import sleep

def calculate_pi(iterations):
    x = (random.random() for i in range(iterations))
    y = (random.random() for i in range(iterations))
    r_squared = [xi**2 + yi**2 for xi, yi in zip(x, y)]
    percent_coverage = sum([r <= 1 for r in r_squared]) / len(r_squared)
    sleep(1)
    return 4 * percent_coverage