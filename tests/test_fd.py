import numpy as np

from mag_diffusion_pinn.fd import fd_ramp, fd_solve
from mag_diffusion_pinn.reference import exact_A_images, exact_S, ramp_response


def test_cn_second_order_on_S():
    errs = []
    for N in (25, 50, 100):
        xi, u = fd_solve(N, N, 0.1, "S")
        errs.append(np.abs(u - exact_S(xi, 0.1)).max())
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(1.8 < r < 2.3 for r in rates), (errs, rates)


def test_fd_A_close_to_exact_at_moderate_tau():
    xi, u = fd_solve(400, 800, 0.1, "A")
    assert np.abs(u - exact_A_images(xi, 0.1)).max() < 1e-3


def test_fd_history_shape():
    xi, t, H = fd_solve(40, 20, 0.1, "S", return_history=True)
    assert H.shape == (21, 41) and t.shape == (21,) and abs(t[-1] - 0.1) < 1e-12


def test_fd_ramp_matches_duhamel_reference():
    xi, t, H = fd_ramp(400, 4000, 0.5)
    k = 2000                                   # t = 0.25
    probe = np.linspace(0.1, 0.9, 9)
    fd = np.array([np.interp(x, xi, H[k]) for x in probe])
    assert np.abs(fd - ramp_response(probe, 0.25)).max() < 2e-4


def test_fd_ramp_with_theta():
    xi, t, H = fd_ramp(400, 4000, 0.5, theta=0.6)
    k = 3200                                   # t = 0.4
    probe = np.array([0.2, 0.5, 0.8])
    fd = np.array([np.interp(x, xi, H[k]) for x in probe])
    assert np.abs(fd - ramp_response(probe, 0.4, theta=0.6)).max() < 2e-4
