import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp


#  Restart-cause codes
NO_RESTART  = 0
COND_A_ONLY = 1   # angle condition (a) failed
COND_B_ONLY = 2   # norm condition (b) failed
COND_BOTH   = 3   # both failed


#  Conjugate direction beta formulas
def beta_FR(g, g_new, d):
    denom = float(jnp.dot(g, g))
    if denom <= 0:
        return 0.0
    return float(jnp.dot(g_new, g_new)) / denom


def beta_PRP_plus(g, g_new, d):
    denom = float(jnp.dot(g, g))
    if denom <= 0:
        return 0.0
    val = float(jnp.dot(g_new, g_new - g)) / denom
    return max(val, 0.0)


def beta_HZ(g, g_new, d):
    y = g_new - g
    dy = float(jnp.dot(d, y))
    if abs(dy) < 1e-12:
        return 0.0
    y_norm_sq = float(jnp.dot(y, y))
    return float(jnp.dot(y - 2.0 * d * (y_norm_sq / dy), g_new)) / dy


BETA_FORMULAS = {
    "FR":   beta_FR,
    "PRP+": beta_PRP_plus,
    "HZ":   beta_HZ,
}


#  Backtracking Armijo line search
def backtracking_armijo(f_vec, f_x, g_x, x, d, eta, theta, max_bt=200):
    gd = float(jnp.dot(g_x, d))
    if gd >= 0:
        return 0.0, f_x

    alpha = 1.0
    for _ in range(max_bt):
        f_trial = float(f_vec(x + alpha * d))
        if f_trial <= f_x + eta * alpha * gd:
            return alpha, f_trial
        alpha *= theta
    return alpha, float(f_vec(x + alpha * d))


#  Hybrid NCG + NCD algorithm with line search
def hybrid_ncg_ncd(x0, f_vec, grad_f, hess_f,
                   L2, sigma, kappa, p, q, eta, theta,
                   eps_g, eps_H,
                   beta_fmla="PRP+",
                   zeta=0.1,
                   max_iter=2000):
    """
    Run Algorithm 1 on the objective f_vec.

    Inputs:
        x0       : initial point (numpy or jax array)
        f_vec    : objective, callable on a flat vector
        grad_f   : gradient, callable on a flat vector
        hess_f   : Hessian, callable on a flat vector
        L2       : Hessian Lipschitz constant (for NCD step length)
        sigma, kappa, p, q : restart-condition parameters
        eta, theta : Armijo line-search parameters
        eps_g, eps_H : termination tolerances on gradient norm and -lam_min
        beta_fmla : "FR", "PRP+", or "HZ"
        zeta     : gradient-norm threshold used to color
                   plots with Regime 1 vs Regime 3
        max_iter : iteration limit

    Returns a dictionary with the trajectory and per-iteration metadata.
    """
    if beta_fmla not in BETA_FORMULAS:
        raise ValueError(f"unknown beta_fmla {beta_fmla!r}; "
                         f"choose from {list(BETA_FORMULAS)}")
    beta_func = BETA_FORMULAS[beta_fmla]

    x = jnp.array(x0, dtype=jnp.float64)
    g = grad_f(x)
    d = -g

    history = {
        "x":             [np.array(x)],
        "f":             [float(f_vec(x))],
        "grad":          [float(jnp.linalg.norm(g))],
        "regime":        [],
        "alpha":         [],
        "restart":       [],
        "restart_cause": [],
        "Lhat_step":     [],
        "beta_fmla":     beta_fmla,
        "beta_value":    [],
    }
    stop_reason = "max_iter"

    for k in range(max_iter):
        g = grad_f(x)
        norm_g = float(jnp.linalg.norm(g))
        f_x = float(f_vec(x))

        H = hess_f(x)
        eigs_H = jnp.linalg.eigvalsh(H)
        history["Lhat_step"].append(float(jnp.max(jnp.abs(eigs_H))))

        if norm_g < eps_g:
            eigvals, eigvecs = jnp.linalg.eigh(H)
            lam_min = float(eigvals[0])

            if -lam_min < eps_H:
                stop_reason = "approx_2nd_order_stationary"
                break

            # NCD step
            v = eigvecs[:, 0]
            if float(jnp.dot(g, v)) > 0:
                v = -v
            x = x + (2.0 * lam_min / L2) * v
            history["regime"].append(2)
            history["alpha"].append(2.0 * lam_min / L2)
            history["restart"].append(False)
            history["restart_cause"].append(NO_RESTART)
            d = -grad_f(x)
            history["beta_value"].append(float("nan"))

        else:
            # NCG step
            alpha, _ = backtracking_armijo(f_vec, f_x, g, x, d, eta, theta)
            x_new = x + alpha * d
            g_new = grad_f(x_new)

            beta = beta_func(g, g_new, d)
            history["beta_value"].append(float(beta))

            d_new = -g_new + beta * d

            norm_g_new = float(jnp.linalg.norm(g_new))
            cond_a = float(jnp.dot(g_new, d_new)) >= -sigma * norm_g_new**(1+p)
            cond_b = float(jnp.linalg.norm(d_new)) >= kappa * norm_g_new**q

            if cond_a and cond_b:
                cause = COND_BOTH
            elif cond_a:
                cause = COND_A_ONLY
            elif cond_b:
                cause = COND_B_ONLY
            else:
                cause = NO_RESTART

            if cond_a or cond_b:
                d_new = -g_new

            x = x_new
            d = d_new

            regime = 1 if (zeta is None or norm_g >= zeta) else 3
            history["regime"].append(regime)
            history["alpha"].append(alpha)
            history["restart"].append(cause != NO_RESTART)
            history["restart_cause"].append(cause)

        history["x"].append(np.array(x))
        history["f"].append(float(f_vec(x)))
        history["grad"].append(float(jnp.linalg.norm(grad_f(x))))

    history["x"] = np.array(history["x"])
    history["f"] = np.array(history["f"])
    history["grad"] = np.array(history["grad"])
    history["regime"] = np.array(history["regime"])
    history["restart"] = np.array(history["restart"])
    history["restart_cause"] = np.array(history["restart_cause"])
    history["Lhat_step"] = np.array(history["Lhat_step"])
    history["stop"] = stop_reason
    history["L1_hat"] = float(np.max(history["Lhat_step"]))
    history["beta_value"] = np.array(history["beta_value"])
    return history


#  Backtracking GD
def gradient_descent(x0, f_vec, grad_f, eta, theta, eps_g, max_iter=2000):
    x = jnp.array(x0, dtype=jnp.float64)

    history = {
        "x":    [np.array(x)],
        "f":    [float(f_vec(x))],
        "grad": [float(jnp.linalg.norm(grad_f(x)))],
        "alpha": [],
    }

    for _ in range(max_iter):
        g = grad_f(x)
        if float(jnp.linalg.norm(g)) < eps_g:
            break
        f_x = float(f_vec(x))
        alpha, _ = backtracking_armijo(f_vec, f_x, g, x, -g, eta, theta)
        x = x + alpha * (-g)
        history["alpha"].append(alpha)
        history["x"].append(np.array(x))
        history["f"].append(float(f_vec(x)))
        history["grad"].append(float(jnp.linalg.norm(grad_f(x))))

    history["x"]    = np.array(history["x"])
    history["f"]    = np.array(history["f"])
    history["grad"] = np.array(history["grad"])
    history["alpha"] = np.array(history["alpha"])
    return history


#  Plotting Functions
COLOR_REG = {1: "C0", 2: "C3", 3: "C2"}
REGIME_LABEL = {1: "regime 1", 2: "regime 2", 3: "regime 3"}

CAUSE_STYLE = {
    COND_A_ONLY: dict(marker="s", facecolors="none", edgecolors="black",
                      s=80, lw=1.2, label="(a) only"),
    COND_B_ONLY: dict(marker="o", facecolors="none", edgecolors="purple",
                      s=80, lw=1.2, label="(b) only"),
    COND_BOTH:   dict(marker="D", facecolors="none", edgecolors="orange",
                      s=80, lw=1.2, label="(a)+(b)"),
}


def _restart_markers(ax, out, y_values):
    cause = out["restart_cause"]
    counts = {c: int(np.sum(cause == c)) for c in CAUSE_STYLE}
    for c, style in CAUSE_STYLE.items():
        idx = np.where(cause == c)[0]
        if len(idx) == 0:
            continue
        label = f"{style['label']} ({counts[c]})"
        kw = {k: v for k, v in style.items() if k != "label"}
        ax.scatter(idx, y_values[idx], label=label, zorder=5, **kw)


def _regime_legend_handles():
    return [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLOR_REG[r], markeredgecolor=COLOR_REG[r],
               markersize=6, label=REGIME_LABEL[r])
        for r in (1, 2, 3)
    ]


def _save_and_close(fig, savepath):
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_subopt(out, savepath, title, f_star):
    delta = np.maximum(out["f"] - f_star, 1e-16)
    K = len(out["regime"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(np.arange(K + 1), delta, "-", color="0.6", lw=0.8, zorder=0)
    for k in range(K):
        ax.scatter(k, delta[k], c=COLOR_REG[int(out["regime"][k])],
                   s=14, zorder=2)
    _restart_markers(ax, out, delta[:K])
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$f(x_k) - f^*$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="major")
    handles, labels = ax.get_legend_handles_labels()
    regime_handles = _regime_legend_handles()
    ax.legend(regime_handles + handles,
              [h.get_label() for h in regime_handles] + labels,
              fontsize=7, loc="upper right")
    _save_and_close(fig, savepath)


def plot_grad(out, savepath, title):
    g = out["grad"]
    K = len(out["regime"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(np.arange(K + 1), g, "-", color="0.6", lw=0.8, zorder=0)
    for k in range(K):
        ax.scatter(k, g[k], c=COLOR_REG[int(out["regime"][k])],
                   s=14, zorder=2)
    _restart_markers(ax, out, g[:K])
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|g_k\|$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="major")
    handles, labels = ax.get_legend_handles_labels()
    regime_handles = _regime_legend_handles()
    ax.legend(regime_handles + handles,
              [h.get_label() for h in regime_handles] + labels,
              fontsize=7, loc="upper right")
    _save_and_close(fig, savepath)


def plot_alpha(out, savepath, title):
    K = len(out["regime"])
    alphas = out["alpha"][:K]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(np.arange(K), alphas, "-", color="0.6", lw=0.8, zorder=0)
    ncg_mask = out["regime"] != 2
    for k in range(K):
        if not ncg_mask[k]:
            continue
        ax.scatter(k, alphas[k], c=COLOR_REG[int(out["regime"][k])],
                   s=14, zorder=2)
    _restart_markers(ax, out, np.array(alphas))
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"line-search stepsize $\alpha_k$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=7, loc="lower right")
    _save_and_close(fig, savepath)


def plot_alpha_compare(runs, savepath, title):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, Rv, out in runs:
        K = len(out["regime"])
        alphas = out["alpha"][:K]
        ax.semilogy(np.arange(K), alphas, lw=1.2,
                    label=f"{label}  ($R={Rv:.2f}$)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"line-search stepsize $\alpha_k$")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="major")
    _save_and_close(fig, savepath)

def plot_beta_compare(runs, savepath, title):
    """Compare beta_k values across configurations (NCG steps only)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, Rv, out in runs:
        b = out["beta_value"]
        K = len(b)
        # Take absolute value for log scale; NaN entries (NCD steps) skipped automatically
        mask = ~np.isnan(b)
        idx = np.arange(K)[mask]
        ax.semilogy(idx, np.maximum(np.abs(b[mask]), 1e-16),
                    lw=1.2, label=f"{label}  ($R={Rv:.2f}$)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"conjugate parameter $|\beta_k|$")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="major")
    _save_and_close(fig, savepath)


def plot_rates_with_gd(runs, gd_out, savepath, title, f_star):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for label, Rv, out in runs:
        delta = np.maximum(out["f"] - f_star, 1e-16)
        n_restart = int(out["restart"].sum())
        ax.semilogy(np.arange(len(delta)), delta, lw=1.4,
                    label=f"{label}  ($R={Rv:.2f}$, restarts={n_restart})")
    gd_delta = np.maximum(gd_out["f"] - f_star, 1e-16)
    ax.semilogy(np.arange(len(gd_delta)), gd_delta,
                "--", lw=2.0, color="black", label="GD (backtracking)")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$f(x_k) - f^*$")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.3, which="major")
    _save_and_close(fig, savepath)


def plot_restart_summary(runs, savepath, title, x_label, x_values):
    n_restart = [int(out["restart"].sum()) for _, _, out in runs]
    n_total   = [len(out["regime"]) for _, _, out in runs]
    n_a    = [int(np.sum(out["restart_cause"] == COND_A_ONLY))
              for _, _, out in runs]
    n_b    = [int(np.sum(out["restart_cause"] == COND_B_ONLY))
              for _, _, out in runs]
    n_both = [int(np.sum(out["restart_cause"] == COND_BOTH))
              for _, _, out in runs]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.6
    positions = np.arange(len(runs))

    ax.bar(positions, n_total, width=width,
           color="0.85", edgecolor="0.4", label="total NCG iterations")
    bottom = np.zeros(len(runs))
    ax.bar(positions, n_a, width=width, bottom=bottom,
           color="C0", alpha=0.85, label="restart cond. (a) only")
    bottom = bottom + np.array(n_a)
    ax.bar(positions, n_b, width=width, bottom=bottom,
           color="C4", alpha=0.85, label="restart cond. (b) only")
    bottom = bottom + np.array(n_b)
    ax.bar(positions, n_both, width=width, bottom=bottom,
           color="C1", alpha=0.85, label="restart (a)+(b)")

    for i, (nt, nr) in enumerate(zip(n_total, n_restart)):
        ax.text(i, nt + 0.5, f"{nr}/{nt}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in x_values], rotation=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("iteration count")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    _save_and_close(fig, savepath)
