"""
Numba-accelerated acceleration calculations.

This module provides JIT-compiled versions of the acceleration calculation
functions for significant speedups (2-5x typical improvement).

Note: We use nopython=True (njit) for maximum performance, but avoid
fastmath=True to maintain numerical accuracy with the NumPy reference.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _smooth_outliers_numba(
    arr: np.ndarray, threshold: float, use_abs: bool
) -> np.ndarray:
    """Vectorized outlier smoothing using forward fill logic (Numba-compiled)."""
    result = arr.copy()
    n = len(arr)
    if n < 3:
        return result

    for i in range(1, n - 1):
        val = result[i]
        if use_abs:
            exceeds = np.abs(val) > threshold
        else:
            exceeds = val > threshold
        if exceeds:
            result[i] = result[i - 1]

    return result


@njit(cache=True)
def _gradient_1d(arr: np.ndarray) -> np.ndarray:
    """Compute gradient of 1D array (Numba-compiled, matches np.gradient edge behavior)."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)

    if n < 2:
        result[:] = 0.0
        return result

    result[0] = arr[1] - arr[0]
    result[n - 1] = arr[n - 1] - arr[n - 2]

    for i in range(1, n - 1):
        result[i] = (arr[i + 1] - arr[i - 1]) / 2.0

    return result


@njit(cache=True)
def _convolve_same(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    1D convolution with uniform kernel and 'same' mode (Numba-compiled).

    Matches np.convolve(arr, np.ones(k)/k, mode='same') exactly.
    """
    n = len(arr)
    k = kernel_size
    kernel_val = 1.0 / k

    full_len = n + k - 1
    full_result = np.zeros(full_len, dtype=np.float64)

    for i in range(n):
        for j in range(k):
            full_result[i + j] += arr[i] * kernel_val

    start = (k - 1) // 2
    result = full_result[start : start + n].copy()

    return result


@njit(cache=True)
def calculate_x_acceleration_numba(
    vx_array: np.ndarray,
    dtime: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Calculate and smooth X-acceleration component (Numba-compiled)."""
    n = len(vx_array)
    dvx = _gradient_1d(vx_array)

    ax = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.abs(dtime[i]) > 1e-15:
            ax[i] = dvx[i] / dtime[i]
        else:
            ax[i] = 0.0

    ax = _smooth_outliers_numba(ax, 25.0, False)
    return _convolve_same(ax, kernel_size)


@njit(cache=True)
def calculate_y_acceleration_numba(
    vx_array: np.ndarray,
    dx: np.ndarray,
    y_array: np.ndarray,
    ds: np.ndarray,
    vx_sq: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """Calculate and smooth Y-acceleration component (Numba-compiled)."""
    n = len(y_array)
    dy = _gradient_1d(y_array)

    eps = np.finfo(np.float64).eps
    theta = np.empty(n, dtype=np.float64)
    for i in range(n):
        theta[i] = np.arctan2(dy[i], dx[i] + eps)

    if n > 1:
        theta[0] = theta[1]

    theta_unwrap = np.empty(n, dtype=np.float64)
    theta_unwrap[0] = theta[0]
    for i in range(1, n):
        diff = theta[i] - theta[i - 1]
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        theta_unwrap[i] = theta_unwrap[i - 1] + diff

    dtheta = _gradient_1d(theta_unwrap)
    dtheta = _smooth_outliers_numba(dtheta, 0.5, True)

    ay = np.empty(n, dtype=np.float64)
    for i in range(n):
        C = dtheta[i] / (ds[i] + 0.0001)
        ay[i] = vx_sq[i] * C
        if np.abs(ay[i]) > 150.0:
            ay[i] = 0.0

    return _convolve_same(ay, kernel_size)


@njit(cache=True)
def calculate_z_acceleration_numba(
    vx_array: np.ndarray,
    dx: np.ndarray,
    z_array: np.ndarray,
    ds: np.ndarray,
    vx_sq: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """Calculate and smooth Z-acceleration component (Numba-compiled)."""
    n = len(z_array)
    dz = _gradient_1d(z_array)

    eps = np.finfo(np.float64).eps
    z_theta = np.empty(n, dtype=np.float64)
    for i in range(n):
        z_theta[i] = np.arctan2(dz[i], dx[i] + eps)

    if n > 1:
        z_theta[0] = z_theta[1]

    z_theta_unwrap = np.empty(n, dtype=np.float64)
    z_theta_unwrap[0] = z_theta[0]
    for i in range(1, n):
        diff = z_theta[i] - z_theta[i - 1]
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        z_theta_unwrap[i] = z_theta_unwrap[i - 1] + diff

    z_dtheta = _gradient_1d(z_theta_unwrap)
    z_dtheta = _smooth_outliers_numba(z_dtheta, 0.5, True)

    az = np.empty(n, dtype=np.float64)
    for i in range(n):
        z_C = z_dtheta[i] / (ds[i] + 0.0001)
        az[i] = vx_sq[i] * z_C
        if np.abs(az[i]) > 150.0:
            az[i] = 0.0

    return _convolve_same(az, kernel_size)


@njit(cache=True)
def calculate_all_accelerations_numba(
    vx_array: np.ndarray,
    time_array: np.ndarray,
    x_array: np.ndarray,
    y_array: np.ndarray,
    z_array: np.ndarray,
    dist_array: np.ndarray,
    nax: int = 3,
    nay: int = 9,
    naz: int = 9,
) -> tuple:
    """
    Calculate all acceleration components in a single optimized call.

    Returns:
        tuple: (ax, ay, az) acceleration arrays
    """
    dtime = _gradient_1d(time_array)
    dx = _gradient_1d(x_array)
    ds = _gradient_1d(dist_array)
    vx_sq = vx_array * vx_array

    ax = calculate_x_acceleration_numba(vx_array, dtime, nax)
    ay = calculate_y_acceleration_numba(vx_array, dx, y_array, ds, vx_sq, nay)
    az = calculate_z_acceleration_numba(vx_array, dx, z_array, ds, vx_sq, naz)

    return ax, ay, az


def warm_up_jit():
    """Warm up JIT compilation with dummy data."""
    n = 100
    vx = np.random.rand(n).astype(np.float64)
    t = np.linspace(0, 10, n).astype(np.float64)
    x = np.random.rand(n).astype(np.float64)
    y = np.random.rand(n).astype(np.float64)
    z = np.random.rand(n).astype(np.float64)
    d = np.linspace(0, 1000, n).astype(np.float64)

    calculate_all_accelerations_numba(vx, t, x, y, z, d)
