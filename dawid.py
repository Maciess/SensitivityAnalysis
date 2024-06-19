import numpy as np
import minterpy as mp
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as npr


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
    const_mult = np.square(2 / np.pi) / variance
    vi = (nwt_diff * nwt_diff).integrate_over() * const_mult * np.power(1/2, spatial_dimension)
    return vi


def compute_dgsm(nwt_poly, variance):
    spatial_dimension = domain.shape[0]
    dgsm_vals = np.zeros(spatial_dimension, dtype=np.float64)
    for i in range(spatial_dimension):
        dgsm_vals[i] = dgsm(nwt_poly, i, variance)
        
    return dgsm_vals
    
    
def visualize(values, ylab):
    plt.bar(["R_b1", "R_b2", "R_f", "R_c1", "R_c2", "beta"], values)
    plt.ylabel(ylab)
    plt.xlabel("OTL parameters")
    plt.show()

    
def run_dgsm():
    variance_value = 1.2
    nwt_poly = get_nwt_interpolant(otl_normalized, spatial_dimension=domain.shape[0], poly_degree=8, lp_degree=1)
    D = compute_dgsm(nwt_poly, variance_value)
    visualize(D, "DGSM")
    

def get_asbm_c_matrix(nwt_poly, spatial_dimension=6):
    gradient = [nwt_poly.partial_diff(i) for i in range(spatial_dimension)]
    C = np.zeros((spatial_dimension, spatial_dimension), dtype=np.float64)
    for i in range(spatial_dimension):
        for j in range(i, spatial_dimension):
            C[i,j] = (gradient[i] * gradient[j]).integrate_over() * np.power(1/2, spatial_dimension)
            print(f"{i}, {j}")
    return C


def get_asbm_eigendecomposition(C):
    v, w = la.eigh(C, UPLO="U")
    return v, w


def asbm(v, w, i, k):
    d = v.shape[0]
    return np.sum([
        v[j] * np.square(w[i, j])
        for j in range(d - k, d)
    ])


def run_asbm():
    spatial_domain = domain.shape[0]
    nwt_poly = get_nwt_interpolant(otl_normalized, spatial_domain, 4, 2.0)
    C = get_asbm_c_matrix(nwt_poly, spatial_domain)
    v, w = get_asbm_eigendecomposition(C)

    asbm_values = np.zeros(spatial_domain, dtype=np.float64)
    for i in range(spatial_domain):
        asbm_values[i] = asbm(v, w, i, 1)
    
    visualize(asbm_values, "Active-subspace-based measures")
    asbm_scatter_plot(w)


def asbm_scatter_plot(w):
    sample = npr.uniform(-1, 1, (1000, 6))
    w1 = w[:,5]
    proj = np.apply_along_axis(lambda x, ww=w1: np.dot(x, ww), axis=1, arr=sample)

    plt.scatter(proj, otl_normalized(sample))
    plt.show()
