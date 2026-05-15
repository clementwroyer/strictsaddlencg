import os
import pickle
import json
import numpy as np
import jax.numpy as jnp

def compute_R(p, q):
    """The asymptotic exponent of the Regime 3 complexity bound."""
    return max(1 + p, 2 * (1 + p - q)) / 2.0


def estimate_gamma(hess_f, x_final, eig_floor=1e-10):
    """
    Empirical strong-convexity constant: smallest positive Hessian
    eigenvalue at x_final. Returns NaN if the Hessian is essentially zero.
    """
    H = hess_f(jnp.array(x_final))
    eigs = np.array(jnp.linalg.eigvalsh(H))
    pos = eigs[eigs > eig_floor]
    return float(np.min(pos)) if len(pos) > 0 else float("nan")


def make_outdir(config):
    """
    Folder name from run config. Includes cn so different condition numbers
    don't overwrite each other.
    """
    n, r = config["dim"]
    parts = [
        "figures",
        config["beta_fmla"].replace("+", "plus"),
        f"sig={config['sigma']}",
        f"kap={config['kappa']}",
        f"dim={n}x{r}",
        f"cn={config['cn']}",
    ]
    return "_".join(parts)


def fname(outdir, stub, p, q, kind):
    """Standardized filename for per-config plots."""
    return os.path.join(outdir, f"{stub}__p={p}__q={q}__{kind}.png")


def sweep_fname(outdir, stub, kind):
    """Standardized filename for sweep-level plots (rates, restarts, alpha)."""
    return os.path.join(outdir, f"{stub}__{kind}.png")

def save_sweep_data(runs, outdir, stub):
    """Save raw run data (pickle) and a summary (txt)."""
    # Pickle: full trajectories for everything
    pkl_path = os.path.join(outdir, f"{stub}__data.pkl")
    pickle_data = {
        "configs": [(label, Rv) for label, Rv, _ in runs],
        "runs":    [out for _, _, out in runs],
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pickle_data, f)

    # Plain-text summary
    txt_path = os.path.join(outdir, f"{stub}__summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'config':>20}  {'R':>6}  {'iters':>6}  {'restarts':>8}  "
                f"{'final f-f*':>12}\n")
        f.write("-" * 70 + "\n")
        for label, Rv, out in runs:
            K = len(out["regime"])
            n_r = int(out["restart"].sum())
            final = float(out["f"][-1])
            f.write(f"{label:>20}  {Rv:>6.3f}  {K:>6d}  {n_r:>8d}  "
                    f"{final:>12.3e}\n")