import numpy as np
import pytest

from mag_diffusion_pinn.reference import (exact_A_images, exact_A_series, exact_S, ramp_g, ramp_gprime,
                                           ramp_reference_grid, ramp_response)


_trapz = getattr(np, 'trapezoid', None) or np.trapz


def pde_residual_fd(f, xi, t, h=1e-3, dt=1e-5, theta=1.0):
    dbdt = (f(xi, t + dt) - f(xi, t - dt)) / (2 * dt)
    d2b = (f(xi + h, t) - 2 * f(xi, t) + f(xi - h, t)) / h ** 2
    return dbdt - theta * d2b


def test_S_satisfies_pde_and_boundaries():
    x = np.linspace(0.1, 0.9, 9)
    assert np.abs(pde_residual_fd(exact_S, x, 0.05)).max() < 1e-4
    assert abs(exact_S(0.0, 0.1)) < 1e-15 and abs(exact_S(1.0, 0.1)) < 1e-12
    assert np.allclose(exact_S(x, 0.0), np.sin(np.pi * x))


@pytest.mark.parametrize("tau", [0.01, 0.05, 0.2])
def test_A_series_and_images_agree(tau):
    xi = np.linspace(0, 1, 201)
    assert np.abs(exact_A_series(xi, tau) - exact_A_images(xi, tau)).max() < 1e-6


def test_A_boundaries_and_pde():
    assert abs(exact_A_images(0.0, 0.05) - 1.0) < 1e-12
    assert abs(exact_A_images(1.0, 0.05)) < 1e-12
    x = np.linspace(0.1, 0.9, 9)
    assert np.abs(pde_residual_fd(exact_A_images, x, 0.05)).max() < 1e-3


def test_ramp_shape():
    assert ramp_g(0.0) == 0 and ramp_g(0.05) == 1 and ramp_g(1.0) == 1
    assert abs(ramp_g(0.025) - 0.5) < 1e-12                       # smoothstep symmetry
    t = np.linspace(0.0, 0.05, 20001)
    assert abs(_trapz(ramp_gprime(t), t) - 1.0) < 1e-6          # integral of g' is the jump of 1


def test_ramp_reference_boundaries_and_late_limit():
    xi = np.linspace(0, 1, 51)
    tau = np.array([0.02, 0.05, 0.3])
    R = ramp_reference_grid(xi, tau, n_gl=200)
    assert np.allclose(R[:, 0], ramp_g(tau), atol=1e-10)
    assert np.abs(R[:, -1]).max() < 1e-10
    # long after the ramp the response is the step response delayed by about half the ramp time
    assert np.abs(R[2] - exact_A_images(xi, 0.3 - 0.025)).max() < 5e-3


def test_ramp_quadrature_converged():
    a = ramp_response(0.4, 0.2, n_gl=100)
    b = ramp_response(0.4, 0.2, n_gl=400)
    assert abs(a - b) < 1e-10


def test_theta_scaling_identity():
    """With a ramp this short relative to the observation time, theta only rescales the time argument of the step response."""
    # not exact for a finite ramp, so check the exact identity that matters: theta=1 reference is a time rescaling of
    # theta=2 with ramp length halved:  b(xi, t; theta, tau_r) = b(xi, theta*t; 1, theta*tau_r)
    a = ramp_response(0.3, 0.2, theta=2.0, tau_r=0.05)
    b = ramp_response(0.3, 0.4, theta=1.0, tau_r=0.10)
    assert abs(a - b) < 1e-10


def test_stored_reference_matches(data_dir):
    import os
    p = os.path.join(data_dir, "ref_R.npy")
    f = os.path.join(data_dir, "reference_fields.npz")
    if not (os.path.exists(p) and os.path.exists(f)):
        pytest.skip("stored reference files not present")
    ref = np.load(f)
    stored = np.load(p)
    xi, tau = ref["xi"][::25], ref["tau"][::25]
    assert np.abs(ramp_reference_grid(xi, tau) - stored[::25, ::25]).max() < 1e-9
