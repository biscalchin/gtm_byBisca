from scipy.spatial import distance
import numpy as np

def k3nerror(X1, X2, k):
    """
    Calculate the k-nearest neighbor normalized error (k3n-error) between two datasets.
    This function measures the discrepancy between each point in one dataset and its k-nearest
    neighbors in the other dataset, normalized by the distances among the k-nearest neighbors
    within the same dataset. It can be used to evaluate the quality of low-dimensional mappings
    (k3n-Z-error) or the accuracy of reconstructed data in high-dimensional space (k3n-X-error).

    Parameters
    ----------
    X1 : numpy.array or pandas.DataFrame
        The first dataset, which can be high-dimensional original data or low-dimensional
        projected data depending on the context.
    X2 : numpy.array or pandas.DataFrame
        The second dataset, which can be the corresponding low-dimensional projected data or
        reconstructed high-dimensional data, respectively.
    k : int
        The number of nearest neighbors to consider for each point.

    Returns
    -------
    k3nerror : float
        The k-nearest neighbor normalized error between X1 and X2. This value represents either
        the k3n-Z-error or the k3n-X-error, depending on the datasets provided.

    Notes
    -----
    The k3n-error is a symmetric measure when X1 and X2 are of the same dimensionality but can
    have different interpretations based on the direction of comparison (X1 to X2 or X2 to X1).
    """

    # Convert inputs to numpy arrays for compatibility with distance calculations
    X1 = np.array(X1)
    X2 = np.array(X2)

    # Compute pairwise distances within each dataset
    X1_dist = distance.cdist(X1, X1)
    X2_dist = distance.cdist(X2, X2)

    # Sort the indices in X1 based on distances to identify nearest neighbors
    X1_sorted_indices = np.argsort(X1_dist, axis=1)

    # Ensure the smallest positive values replace zeros in X2 distances to avoid division by zero
    for i in range(X2.shape[0]):
        _replace_zero_with_the_smallest_positive_values(X2_dist[i, :])

    # Exclude the first column (self-distance) and select k nearest neighbors in X1 and X2
    I = np.eye(len(X1_dist), dtype=bool)
    neighbor_dist_in_X1 = np.sort(X2_dist[:, X1_sorted_indices[:, 1:k+1]][I])
    neighbor_dist_in_X2 = np.sort(X2_dist)[:, 1:k+1]

    # Compute the normalized difference between distances in X1's neighbors and X2's neighbors
    sum_k3nerror = ((neighbor_dist_in_X1 - neighbor_dist_in_X2) / neighbor_dist_in_X2).sum()

    # Normalize the total error by the number of points and the number of neighbors to get the average error
    return sum_k3nerror / X1.shape[0] / k


def _replace_zero_with_the_smallest_positive_values(arr):
    """
    Replace all zero values in the given array with the smallest positive non-zero value present in the array.
    This function is particularly useful for preprocessing data before performing operations that cannot handle
    zero values, such as computing logarithms or inverse transformations where zeros would lead to undefined or
    infinite results.

    Parameters
    ----------
    arr : numpy.array
        The array in which zero values are to be replaced. This array is modified in place.

    Notes
    -----
    The function directly modifies the input array (`arr`) and does not return a value. Care should be taken
    when using this function as it changes the original data. This is typically used in preprocessing steps
    where zeros represent missing or undefined values that could otherwise interfere with mathematical operations.
    """

    # Find the smallest positive non-zero value in the array
    smallest_positive_value = np.min(arr[arr != 0])

    # Replace zero values with this smallest positive value
    arr[arr == 0] = smallest_positive_value
