import numpy as np
import minterpy as mp

def draw_sample(sample_size):
    r_b1 = np.random.uniform(-1, 1, size=sample_size)
    r_b2 = np.random.uniform(-1, 1, size=sample_size)
    r_f = np.random.uniform(-1, 1, size=sample_size)
    r_c1 = np.random.uniform(-1, 1, size=sample_size)
    r_c2 = np.random.uniform(-1,  1, size=sample_size)
    beta = np.random.uniform(-1, 1, size=sample_size)
    return np.column_stack((r_b1, r_b2, r_f, r_c1, r_c2, beta))


print(draw_sample(300))