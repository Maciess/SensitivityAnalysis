import numpy as np
from scipy.special import eval_legendre
from typing import Union

def compute_legendre_coefficients(
    multi_index: np.ndarray,
    unisolvent_nodes: np.ndarray,
    lag_coeffs: np.ndarray,
) -> np.ndarray:
    """Compute the Legendre coefficients from an interpolation model in the Lagrange form.
    
    Parameters
    ----------
    multi_index: np.ndarray
        The exponents in a multi-index set.
    unisolvent_nodes: np.ndarray
        The unisolvent nodes associated with the multi-index set.
    lag_coeffs: np.ndarray
        The coefficients of the interpolating Lagrange polynomial,
        i.e., function values evaluated at the unisolvent nodes.
    
    Return
    ------
    np.ndarray
        The orthonormal Legendre coefficients.
    """
    
    # Create a matrix of Legendre polynomial values evaluated at the unisolvent nodes
    xx = np.ones((unisolvent_nodes.shape[0], multi_index.shape[0]))
    
    for i in range(1, xx.shape[1]):
        for j in range(multi_index.shape[1]):
            # Compute the Legendre basis at the unisolvent nodes
            xx[:, i] *= eval_legendre(multi_index[i,j], unisolvent_nodes[:,j])
            # Normalize with the magnitude of the polynomial -> Orthonormal Legendre
            xx[:, i] *= np.sqrt((2 * multi_index[i,j] + 1)) 

    # Compute the coefficients
    legendre_coeffs = np.linalg.inv(xx) @ lag_coeffs
    
    return legendre_coeffs


def compute_sobol_indices(
    multi_index: np.ndarray,
    legendre_coeffs: np.ndarray,
    order: Union[str, int] = "total",
) -> np.ndarray:
    """Compute the Sobol' indices given multi-index set and orthonormal Legendre coeffs.
    
    Parameters
    ----------
    multi_index : np.ndarray
        The exponents in a multi-index set.
    legendre_coeffs : np.ndarray
        The orthonormal Legendre coefficients correspond to
        the multi-index set.
    order : Union[str, int], optional
        The order of the Sobol' indices.
        Default is set to "total"
    
    Returns
    -------
    np.ndarray
        The Sobol' indices of the specified order,
        a one-dimensional array of length ``m``.
        or a two-dimensional array of shape (``m-by-m``).
    """
    mi = multi_index[1:]
    leg_coeffs = legendre_coeffs[1:]
    m = mi.shape[1]
    sobol_indices = np.empty(m)
    yy_var = np.sum(leg_coeffs**2)
    if order == 1:
        for i in range(m):
            # Elements whose i-th entry is non-zero and the rest is zero
            idx = np.where(np.all(np.delete(mi, i, axis=1) == 0, axis=1))
            sobol_indices[i] = np.sum(leg_coeffs[idx]**2) / yy_var
    elif order == "total":
        for i in range(m):
            # Elements whose i-th entry is non-zero
            idx = np.where(mi[:,i] != 0)
            sobol_indices[i] = np.sum(leg_coeffs[idx]**2) / yy_var
    elif order == 2:
        # Pair elements
        sobol_indices = np.empty((m, m))
        for i in range(m):
            for j in range(i,m):
                idx1 = np.all(np.delete(mi, [i,j], axis=1) == 0, axis=1)
                idx2 = mi[:,i] != 0
                idx3 = mi[:,j] != 0
                idx = np.where(np.logical_and(np.logical_and(idx1, idx2), idx3))

                sobol_indices[i,j] = np.sum(leg_coeffs[idx]**2) / yy_var
                sobol_indices[j,i] = sobol_indices[i,j]
            
    return sobol_indices
