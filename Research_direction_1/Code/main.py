import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, hessian, jit

import functions as fn
from utils import *

jax.config.update("jax_enable_x64", True)

# Generic matrix recovery problem
def make_problem(n, r, cn=1.0):
    """
    f(X) = 0.5 * || X X^T - M ||_F^2

    cn = 1.0  -> M = I_n  (well-conditioned, original problem)
    cn > 1.0  -> M = diag of geometric sequence from 1 down to 1/cn
                 (condition number of M is `cn`)
    """
    if cn == 1.0:
        M = jnp.eye(n)
    else:
        diag = jnp.linspace(1.0, 1.0/cn, n)
        M = jnp.diag(diag)

    def f_mat(X):
        return 0.5 * jnp.sum((X @ X.T - M) ** 2)

    def f_vec(x):
        return f_mat(x.reshape(n, r))

    grad_f = jit(grad(f_vec))
    hess_f = jit(hessian(f_vec))

    f_star = 0.0

    return f_vec, grad_f, hess_f, f_star


#  Functions to run Sweeps
def run_one(p, q, x0, problem, config):
    f_vec, grad_f, hess_f, _ = problem
    return fn.hybrid_ncg_ncd(
        x0,
        f_vec=f_vec, grad_f=grad_f, hess_f=hess_f,
        L2=config["L2"], sigma=config["sigma"], kappa=config["kappa"],
        p=p, q=q,
        eta=config["eta"], theta=config["theta"],
        eps_g=config["eps_g"], eps_H=config["eps_H"],
        beta_fmla=config["beta_fmla"],
        zeta=config["zeta"],
        max_iter=config["max_iter"],
    )


def run_sweep(stub, configs, x0, problem, config, outdir,
              sweep_label, x_axis_label, x_axis_values):
    f_vec, grad_f, hess_f, f_star = problem
    runs = []

    print(f"\n[{stub}]   beta = {config['beta_fmla']}, "
          f"sigma = {config['sigma']}, kappa = {config['kappa']}, "
          f"cn = {config['cn']}")
    print(f"{'p':>5} {'q':>5} {'R':>6}  "
          f"{'iters':>5} {'restart':>7}  {'(a)':>4} {'(b)':>4} {'a+b':>4}  "
          f"{'gamma':>7} {'L1':>7}")
    print("-" * 80)

    for (p, q) in configs:
        Rv = compute_R(p, q)
        out = run_one(p, q, x0, problem, config)
        K = len(out["regime"])
        n_r    = int(out["restart"].sum())
        n_a    = int(np.sum(out["restart_cause"] == fn.COND_A_ONLY))
        n_b    = int(np.sum(out["restart_cause"] == fn.COND_B_ONLY))
        n_both = int(np.sum(out["restart_cause"] == fn.COND_BOTH))

        gamma_hat = estimate_gamma(hess_f, out["x"][-1])
        print(f"{p:>5.2f} {q:>5.2f} {Rv:>6.3f}  "
              f"{K:>5d} {n_r:>7d}  {n_a:>4d} {n_b:>4d} {n_both:>4d}  "
              f"{gamma_hat:>7.3f} {out['L1_hat']:>7.3f}")

        # No per-config plots anymore
        runs.append((f"$p={p}$, $q={q}$", Rv, out))

    # GD baseline
    gd_out = fn.gradient_descent(
        x0, f_vec=f_vec, grad_f=grad_f,
        eta=config["eta"], theta=config["theta"],
        eps_g=config["eps_g"], max_iter=config["max_iter"],
    )

    # Summary plots
    fn.plot_rates_with_gd(
        runs, gd_out,
        sweep_fname(outdir, stub, "rates"),
        f"{sweep_label}: rate comparison",
        f_star,
    )
    fn.plot_restart_summary(
        runs,
        sweep_fname(outdir, stub, "restarts"),
        f"{sweep_label}: restart frequency by cause",
        x_axis_label, x_axis_values,
    )
    fn.plot_alpha_compare(
        runs,
        sweep_fname(outdir, stub, "alpha"),
        f"{sweep_label}: line-search stepsize",
    )
    fn.plot_beta_compare(
        runs,
        sweep_fname(outdir, stub, "beta"),
        f"{sweep_label}: conjugate parameter $|\\beta_k|$",
    )

    # Raw data + text summary
    save_sweep_data(runs, outdir, stub)


#  Function for all sweps per config
def run_all_sweeps(x0, problem, config, outdir):
    extreme_configs = [
        (0.0, 0.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (2.0, 1.5),
        (5.0, 3.0),
        (5.0, 10.0),
        (10.0, 5.0),
    ]
    p_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    q_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    run_sweep("exp1_extremes", extreme_configs, x0, problem, config, outdir,
              "Experiment 1: extreme $(p, q)$",
              "$(p, q)$",
              [f"({p},{q})" for (p, q) in extreme_configs])

    run_sweep("exp2a_q1_vary_p", [(p, 1.0) for p in p_values],
              x0, problem, config, outdir,
              "Experiment 2a: $q=1$, varying $p$",
              "$p$", p_values)

    run_sweep("exp2b_q2_vary_p", [(p, 2.0) for p in p_values],
              x0, problem, config, outdir,
              "Experiment 2b: $q=2$, varying $p$",
              "$p$", p_values)

    run_sweep("exp3a_p1_vary_q", [(1.0, q) for q in q_values],
              x0, problem, config, outdir,
              "Experiment 3a: $p=1$, varying $q$",
              "$q$", q_values)

    run_sweep("exp3b_p2_vary_q", [(2.0, q) for q in q_values],
              x0, problem, config, outdir,
              "Experiment 3b: $p=2$, varying $q$",
              "$q$", q_values)


#  Default config
BASE_CONFIG = dict(
    L2       = 20.0,
    sigma    = 0.1,
    kappa    = 2.0,
    eta      = 1e-2,
    theta    = 0.5,
    eps_g    = 1e-7,
    eps_H    = 1e-3,
    zeta     = 0.1,
    max_iter = 3000,
    beta_fmla = "PRP+",
    dim = (5, 5),
    cn = 1.0, # Condition number
)


# SWEEP CONFIG FOR PAPER
if __name__ == "__main__":
    # for dim in [(5, 5), (5, 1)]:
    rng = np.random.default_rng(10)
    n, r = BASE_CONFIG["dim"]
    x0 = 0.05 * rng.standard_normal(n * r)
    for cn in [1, 10, 100, 1000]:
        for sigma, kappa in [(0.1, 2.0), (0.01, 100.0)]:
            for beta in ["PRP+", "HZ", "FR"]:
                config = dict(BASE_CONFIG)
                # config["dim"]       = dim
                config["sigma"]     = sigma
                config["kappa"]     = kappa
                config["beta_fmla"] = beta
                config["cn"] = cn

                problem = make_problem(n, r, cn=cn)
                outdir = make_outdir(config)
                os.makedirs(outdir, exist_ok=True)

                print(f"\n>>> {outdir}")
                run_all_sweeps(x0, problem, config, outdir)