import numpy as np

def draw_sample(sample_size):
    r_b1 = np.random.uniform(-1, 1, size=sample_size)
    r_b2 = np.random.uniform(-1, 1, size=sample_size)
    r_f = np.random.uniform(-1, 1, size=sample_size)
    r_c1 = np.random.uniform(-1, 1, size=sample_size)
    r_c2 = np.random.uniform(-1,  1, size=sample_size)
    beta = np.random.uniform(-1, 1, size=sample_size)
    return np.column_stack((r_b1, r_b2, r_f, r_c1, r_c2, beta))


sample = draw_sample(1000)
print(sample.shape)