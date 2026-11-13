"""Physics-informed neural networks for 1D resistive magnetic diffusion.

Modules
-------
reference : analytic solutions (problems S and A) and the Duhamel reference for the smooth ramp (problem R)
fd        : Crank-Nicolson finite-difference solvers
pinn      : forward PINN (plain, hard constraints, loss weighting, residual-adaptive sampling, causal training)
inverse   : recovering the diffusivity ratio theta from probe data (PINN, least squares, Cramer-Rao bound)
"""
__version__ = "0.1.0"
