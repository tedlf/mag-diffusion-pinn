# Specification: Physics-Informed Neural Network for 1D Magnetic Diffusion

Status: draft v0.1. Owner: Ted Larson Freeman.

## 1. Purpose

Build a small, well-validated physics-informed neural network (PINN) for resistive magnetic diffusion in one dimension. The project has two goals:

1. Learn the practical behavior of PINNs (what trains easily, what fails, and why) on a problem with a known exact answer.
2. Produce a clean, reproducible public repository that demonstrates validated physics-informed ML on a problem relevant to pulsed-power and plasma experiments (field penetration into a conductor, and inference of material properties from probe measurements).

Success is judged by comparison against independent baselines, not by how good a loss curve looks.

## 2. Physics

Resistive diffusion of a magnetic field component B(x, t) through a uniform conductor with conductivity sigma and permeability mu0:

    dB/dt = eta * d2B/dx2,    eta = 1 / (mu0 * sigma)

eta is the magnetic diffusivity (m^2/s). Domain: 0 <= x <= L.

### 2.1 Nondimensionalization

Use b = B / B0, xi = x / L, tau = eta * t / L^2. Then:

    db/dtau = d2b/dxi2,    0 < xi < 1,  tau > 0

All code works in these nondimensional variables. A helper converts to physical units (eta, L, B0) for plots and reporting.

### 2.2 Problem variants

| ID | Initial condition b(xi, 0) | Boundary conditions | Purpose |
|----|---------------------------|---------------------|---------|
| S  | sin(pi * xi) | b(0,tau) = b(1,tau) = 0 | Smooth sanity case. Exact: sin(pi xi) exp(-pi^2 tau). |
| A  | 0 | b(0,tau) = 1, b(1,tau) = 0 | Applied field step at xi = 0 (canonical field-penetration problem). Initial and boundary data are discontinuous at the corner, which is a deliberate difficulty. |
| R  | 0 | b(0,tau) = smooth ramp from 0 to 1 over tau_r, b(1,tau) = 0 | Regularized version of A, used if A proves too stiff for a plain PINN. |

Exact solution for A (series):

    b(xi, tau) = 1 - xi - sum_{n>=1} (2 / (n pi)) sin(n pi xi) exp(-n^2 pi^2 tau)

Independent small-tau form (method of images), used as a cross-check:

    b(xi, tau) = sum_{k>=0} [ erfc((2k + xi) / (2 sqrt(tau))) - erfc((2k + 2 - xi) / (2 sqrt(tau))) ]

Evaluation window: tau in [0, tau_max], with tau_max = 0.5 by default (the field has essentially reached steady state by then).

## 3. Baselines (independent of the PINN)

1. **Analytic solutions** above, implemented and tested first.
2. **Finite-difference solver:** second-order central differences in space and Crank-Nicolson in time, with an optional backward-Euler start-up (Rannacher smoothing). Notebook 01 tests whether problem A needs the start-up; it does not (no oscillations, same accuracy), so plain CN is the default.
3. **Verification of the baselines:** the FD solver must show second-order convergence on variant S, and agree with the analytic solution on variant A to a tolerance that is measured and reported.

The PINN is never validated against itself. Every reported error is against an analytic or FD reference.

## 4. PINN design (initial choices, to be varied in experiments)

- Network: fully connected, input (xi, tau), output b. Start with 4 hidden layers x 64 units, tanh activation. Float32 first; try float64 if precision limits show up.
- Loss: weighted sum of
  - PDE residual at interior collocation points, computed with autograd;
  - boundary condition mismatch;
  - initial condition mismatch.
- Optimizer: Adam, followed by L-BFGS refinement.
- Collocation points: uniform random resampled each epoch; later, residual-adaptive sampling.
- Seeds fixed and recorded. Results reported over several seeds, not one.

### 4.1 Planned variations (each compared against the same baselines)

1. Plain PINN with equal loss weights (expected to struggle on variant A; record how it fails).
2. Loss weighting (manual, then adaptive).
3. Hard-coded boundary and initial conditions through an output transformation.
4. Residual-adaptive collocation sampling.
5. Causal training in tau (earlier times weighted first).
6. Architecture and activation variations (width, depth, Fourier features), only if justified by what the earlier steps show.

## 5. Metrics

- Relative L2 error of b against the reference on a fixed evaluation grid (e.g. 201 x 201 in (xi, tau)).
- Maximum absolute error and its location.
- Mean squared PDE residual on a held-out set of points.
- Wall-clock training time and number of optimizer steps.
- For multi-seed runs: median, min, and max of each metric.

## 6. Milestones and acceptance criteria

**M1: baselines (notebook 01).** Analytic solutions verified against each other and against the PDE; FD solver converges at second order on S; error of FD against A reported. *Done when:* the notebook runs top to bottom with no errors and the convergence plot and table are in place.

**M2: plain PINN on S, then A (notebook 02).** *Done when:* relative L2 error on S is below 1e-3, and the behavior on A (including failures) is documented with plots.

**M3: improvements on A (notebook 03).** Variations in section 4.1, compared on equal terms. *Done when:* the best configuration reaches a relative L2 error on A that is stated and reproducible across seeds, and each change's effect is quantified.

*M3 result (added after the run):* a third variant, R, was added: the boundary value ramps smoothly, b(0, τ) = g(τ), with g a quintic smoothstep over τ_r = 0.05, zero initial condition and b(1, τ) = 0. Its reference is a Duhamel integral over the step solution, cross-checked by finite differences. Over six seeds, hard constraints reach a median relative L2 error of 1.1e-4 on R; on A the best configurations reach about 3.5e-3 (plain: 1.4e-2). See notebook 03 for caveats.

**M4: inverse problem (notebook 04).** Infer eta from sparse, noisy simulated probe measurements of b at a few positions. *Done when:* eta is recovered with a reported error, and sensitivity to noise level and probe placement is shown.

*M4 result (added after the run):* the inverse problem was set up with the diffusivity ratio theta = eta/eta_ref as the unknown in b_t = theta b_xixi (problem-R drive, hard constraints), true theta = 0.6, PINN start 1.0, probe data at 20 times in (0, 0.5], Gaussian noise. Baselines: nonlinear least squares on the exact forward model and the Cramer-Rao bound. See notebook 04 for results and caveats.

**M5: repository structure.** Refactor notebooks into an installable package (`mag_diffusion_pinn`) with tests (analytic solutions, FD convergence, a fast PINN smoke test), a README with results, and a pinned environment. *Done when:* `pytest` passes from a fresh clone and the README reproduces the headline figures.

**Stretch (M6):** a 2D case (for example diffusion into a rectangular conductor, or a circuit-coupled boundary condition in 1D), run on rented compute if needed.

## 7. Repository layout (target)

    mag-diffusion-pinn/
      README.md
      SPEC.md
      requirements.txt
      notebooks/
        01_baselines.ipynb
        02_plain_pinn.ipynb
        03_improvements.ipynb
        04_inverse.ipynb
      src/mag_diffusion_pinn/      (reference, fd, pinn, inverse)
      tests/
      scripts/reproduce_headlines.py

## 8. Risks and open questions

- Discontinuity at the corner of variant A makes pointwise PDE and boundary losses conflict; expect poor accuracy near xi = 0 at small tau.
- Loss balancing between residual, boundary, and initial terms is the usual failure mode; track each term separately.
- Float32 may limit the achievable error floor; test float64 early.
- Reference quality: the FD solution is a check on the analytic forms, not a replacement for them; keep the analytic solution as the arbiter, and report FD error alongside PINN error.

## 9. Out of scope (for now)

Nonlinear (temperature-dependent) resistivity, plasma-specific physics, 3D geometries, and anything requiring experimental data.
