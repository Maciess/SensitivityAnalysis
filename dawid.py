import numpy as np
import minterpy as mp


domain = np.array([
    [50, 150],
    [25, 70],
    [0.5, 3],
    [1.2, 2.5],
    [0.25, 1.2],
    [50, 300],
])


def otl(Rb1, Rb2, Rf, Rc1, Rc2, beta):

    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    
    bR9 = beta * (Rc2 + 9)
    
    w1 = 0.74
    w2 = 11.35
    
    s1 = (Vb1 + w1) * bR9 / (bR9 + Rf)
    s2 = w2 * Rf / (bR9 + Rf)
    s3 = w1 * Rf * bR9 / ((bR9 + Rf) * Rc1)
    
    return s1 + s2 + s3


def otl_normalized(xx):
    params = [translate_interval(domain[i][0], domain[i][1], xx[:,i]) for i in range(domain.shape[0])]
    return otl(*params)
    
    
def translate_interval(a, b, xx):
    avg = (a + b) / 2
    scale = b - avg
    
    return xx * scale + avg


def get_nwt_interpolant(f, spatial_dimension=6, poly_degree=8, lp_degree=1.0):
    
    mi = mp.MultiIndexSet.from_degree(
        spatial_dimension=spatial_dimension,
        poly_degree=poly_degree,
        lp_degree=lp_degree
    )

    grd = mp.Grid(mi)
    lag_coeffs = f(grd.unisolvent_nodes)

    lag_poly = mp.LagrangePolynomial(
        multi_index=mi,
        coeffs=lag_coeffs,
    )
    
    return mp.LagrangeToNewton(lag_poly)()
    
    
def dgsm(nwt_poly, i, variance, spatial_dimension=6):
    nwt_diff = nwt_poly.partial_diff(i)
    interval_length = domain[i][1] - domain[i][0]
    diff_const = np.square(interval_length / 2)
    vi = (nwt_diff * nwt_diff).integrate_over() * diff_const * np.power(1/2, spatial_dimension)
    dgsm_const = np.square(interval_length / np.pi) / variance
    return dgsm_const * vi


def compute_dgsm(nwt_poly, variance):
    spatial_dimension = domain.shape[0]
    dgsm_vals = np.zeros(spatial_dimension, dtype=np.float64)
    for i in range(spatial_dimension):
        dgsm_vals[i] = dgsm(nwt_poly, i, variance)
        
    return dgsm_vals
    
    
def visualize_dgsm(dgsm_values):
    plt.bar(["R_b1", "R_b2", "R_f", "R_c1", "R_c2", "beta"], dgsm_values)
    plt.ylabel("DGSM")
    plt.xlabel("OTL parameters")
    plt.yscale("log")
    plt.show()
    
def run_dgsm():
    variance_value = 0.6095304683143756
    nwt_poly = get_nwt_interpolant(otl_normalized, spatial_dimension=domain.shape[0], poly_degree=8, lp_degree=1)
    D = compute_dgsm(nwt_poly, variance_value)
    visualize_dgsm(D)
