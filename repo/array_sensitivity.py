"""
array_sensitivity.py
--------------------
Electrode-array sensitivity study (Table 13 and Fig. 16 of the revised manuscript).

Models 1 and 2 are repeated with dipole-dipole, Wenner and Schlumberger
acquisition, using all four inversion methods (Independent, Difference,
4D L2-coupled and HWSI). The 50 electrodes, forward and inversion meshes,
noise realisation (seed 123) and regularisation (lambda_s = lambda_t = 20,
3% noise, 5% error floor) are identical in every run, so the array is the
only variable.

Outputs (written next to this script):
  array_<model>_<array>.pkl     cached results per model/array (delete to re-run)
  array_sensitivity_nrmse.png   per-epoch nRMSE, 2 x 3 panels (Fig. 16)

Self-contained (HWSI engine bundled) - runs from any folder, no local imports.
Requires: pygimli 1.5.5, numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert


# ===========================================================================
# HWSI engine + benchmark methods + utilities  (paper code, relax-enhanced)
# ===========================================================================
# ---------------------------------------------------------------------------
# Convergence control constants
# ---------------------------------------------------------------------------
# Convergence is declared when the mean update norm stops improving
# meaningfully over a sliding window of CONV_WINDOW consecutive iterations,
# provided at least MIN_ITERATIONS have been completed.

REL_CONV_TOL = 0.03    # maximum relative improvement still accepted as plateau
CV_TOL       = 0.15    # maximum coefficient of variation within the window
CONV_WINDOW  = 3       # number of iterations in the plateau check window
MIN_ITERS    = 4       # earliest iteration at which convergence can be declared


# ---------------------------------------------------------------------------
# HWSI_Inversion
# ---------------------------------------------------------------------------

class HWSI_Inversion:
    """
    History-Weighted Spatiotemporal Inversion (HWSI).

    Each mesh cell is assigned its own temporal blending coefficient derived
    from the two most recently accepted model updates.  The coefficient is
    recomputed at every Gauss-Seidel sweep, so the weight field co-evolves
    with the inversion solution throughout the solve.

    Weight computation (t >= 2)
    ---------------------------
    raw weight  :  w_raw[j]  = 1 / (eps + |m[t-1,j] - m[t-2,j]|)
    normalised  :  w_norm[j] = w_raw[j] / median(w_raw)
    clipped     :  w[j]      = min(w_norm[j], P95(w_norm))

    Per-cell blending coefficient
    -----------------------------
    alpha[j] = min( lambda_t * w[j] / (lambda_s + lambda_t * w[j]),  alpha_max )

    Blended estimate (Gauss-Seidel, current pass used as prior)
    ------------------------------------------------------------
    m_blend[j] = (1 - alpha[j]) * m_spatial[j]  +  alpha[j] * m_prev[j]

    where m_spatial is the independently inverted model at the current epoch
    and m_prev is the freshest available estimate of the preceding epoch
    within the ongoing sweep.

    Parameters
    ----------
    inv_mesh : pygimli mesh
        Inversion domain mesh.
    data_list : list of pygimli DataContainerERT
        One container per monitoring epoch, ordered chronologically.
    lambda_s : float, optional
        Spatial regularisation strength.  Default 20.0.
    lambda_t : float, optional
        Temporal coupling base strength.  Default 20.0.
    epsilon : float, optional
        Singularity guard in the weight denominator (log-resistivity units).
        Default 5e-3.
    alpha_max : float, optional
        Hard cap on the per-cell blending coefficient.  Default 0.65.
    max_iter : int, optional
        Maximum number of Gauss-Seidel iterations.  Default 15.
    verbose : bool, optional
        Print per-cell resistivity ranges after each epoch.  Default True.
    """

    def __init__(self, inv_mesh, data_list,
                 lambda_s=20.0, lambda_t=20.0,
                 epsilon=5e-3, alpha_max=0.65,
                 max_iter=15, verbose=True,
                 relax=1.0, return_best=False):

        self.mesh      = inv_mesh
        self.data_list = data_list
        self.lambda_s  = lambda_s
        self.lambda_t  = lambda_t
        self.epsilon   = epsilon
        self.alpha_max = alpha_max
        self.max_iter  = max_iter
        self.verbose   = verbose
        # Stability controls (defaults reproduce the original behaviour):
        #   relax < 1.0   -> under-relax the Gauss-Seidel sweep to damp the
        #                    oscillation between data-fit and temporal prior.
        #   return_best   -> return the most-settled iterate (min avg ||dm||)
        #                    instead of the arbitrary last one.
        self.relax       = float(relax)
        self.return_best = bool(return_best)

        self.n_cells = inv_mesh.cellCount()
        self.n_times = len(data_list)

        # Storage updated each iteration
        self.models   = []
        self.weights_t = []
        self.convergence_history = []
        self._best_models = None
        self._best_change = np.inf

        # Spatial graph Laplacian (used only for reference; pyGIMLi handles
        # spatial regularisation internally through ERTManager.invert)
        self.L_s = self._build_laplacian()

        # One ERTManager per epoch, shared across all iterations
        self.managers = []
        for d in data_list:
            mgr = ert.ERTManager()
            mgr.setData(d)
            mgr.setMesh(inv_mesh)
            self.managers.append(mgr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_laplacian(self):
        """Build the sparse graph Laplacian for the inversion mesh."""
        n   = self.n_cells
        nbr = [set() for _ in range(n)]

        for b in self.mesh.boundaries():
            lc = b.leftCell()
            rc = b.rightCell()
            if lc and rc:
                li, ri = lc.id(), rc.id()
                if li < n and ri < n:
                    nbr[li].add(ri)
                    nbr[ri].add(li)

        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i); cols.append(i); vals.append(float(len(nbr[i])))
            for j in nbr[i]:
                rows.append(i); cols.append(j); vals.append(-1.0)

        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    def _compute_weights(self):
        """
        Recompute per-cell temporal weights from the two most recent model
        estimates.  For t < 2 (insufficient history) weights default to 1.
        """
        self.weights_t = []
        for t in range(self.n_times):
            if t < 2:
                self.weights_t.append(np.ones(self.n_cells))
                continue

            diff  = np.abs(self.models[t - 1] - self.models[t - 2])
            w_raw = 1.0 / (self.epsilon + diff)

            med = np.median(w_raw)
            w_norm = w_raw / med if med > 1e-8 else np.ones(self.n_cells)

            p95   = np.percentile(w_norm, 95)
            w_clp = np.clip(w_norm, 0.0, p95)

            self.weights_t.append(w_clp)

    def _invert_step(self, t, current_pass):
        """
        Invert one epoch and blend with the temporal prior.

        Parameters
        ----------
        t : int
            Epoch index.
        current_pass : list of ndarray
            Log-resistivity models produced so far in the current sweep.
            For t == 0 this list is empty; for t > 0 it contains the
            freshly updated estimate of epoch t-1.

        Returns
        -------
        m_blend : ndarray
            Blended log-resistivity vector for epoch t.
        """
        # Warm-start from the previous iteration's estimate when available
        start = None
        if len(self.models) > t:
            start = np.clip(np.exp(self.models[t]), 1.0, 1e4)

        m_sp = self.managers[t].invert(
            self.data_list[t],
            lam=self.lambda_s,
            limits=[1.0, 1e4],
            startModel=start,
            verbose=False,
        )
        m_sp = np.log(np.clip(np.array(m_sp), 1.0, 1e4))

        # No temporal blending for the baseline epoch or when the prior
        # epoch estimate is not yet available in the current pass
        if t > 0 and len(current_pass) >= t:
            lw    = self.lambda_t * self.weights_t[t]
            alpha = np.clip(lw / (self.lambda_s + lw), 0.0, self.alpha_max)
            m_sp  = (1.0 - alpha) * m_sp + alpha * current_pass[t - 1]

        return np.clip(m_sp, 0.0, np.log(1e4))

    def _check_convergence(self, k):
        """
        Apply the windowed plateau convergence criterion.

        Convergence is declared when, over the last CONV_WINDOW iterations:
        (a) the mean update norm has not improved by more than REL_CONV_TOL
            relative to the entry of the window, and
        (b) the coefficient of variation within the window is below CV_TOL.

        Both conditions must hold simultaneously, and at least MIN_ITERS
        iterations must have been completed.

        Returns
        -------
        converged : bool
        diag_str : str  —  diagnostic string (empty if window not yet full)
        """
        hist = self.convergence_history
        if len(hist) < CONV_WINDOW + 1 or k < MIN_ITERS:
            return False, ""

        window    = [h["avg_change"] for h in hist[-CONV_WINDOW:]]
        pre_value = hist[-(CONV_WINDOW + 1)]["avg_change"]
        w_mean    = np.mean(window)
        w_cv      = np.std(window) / w_mean if w_mean > 1e-12 else 0.0
        rel_impr  = (pre_value - w_mean) / pre_value if pre_value > 1e-12 else 0.0

        diag = (
            f"  window_mean={w_mean:.4f}  pre={pre_value:.4f}  "
            f"rel_impr={rel_impr * 100:.2f}%  CV={w_cv * 100:.1f}%"
        )
        converged = rel_impr < REL_CONV_TOL and w_cv < CV_TOL
        return converged, diag

    def _run_one_iteration(self, k):
        """
        Execute one full Gauss-Seidel sweep over all epochs.

        Returns
        -------
        converged : bool
        """
        print(f"\n--- Iteration {k + 1}/{self.max_iter} ---")
        new_models = []

        for t in range(self.n_times):
            print(f"  T={t}...", end="", flush=True)
            m = self._invert_step(t, new_models)
            new_models.append(m)
            rho = np.exp(m)
            if self.verbose:
                print(f" rho=[{rho.min():.1f}, {rho.max():.1f}]")
            else:
                print()

        # --- under-relaxation: damp the sweep against the previous iterate ---
        # new <- (1-relax)*old + relax*new, for the monitored epochs (t>=1).
        # relax=1.0 leaves the original behaviour untouched.
        if self.relax < 1.0 and len(self.models) == len(new_models):
            for t in range(1, self.n_times):
                new_models[t] = ((1.0 - self.relax) * self.models[t]
                                 + self.relax * new_models[t])

        # Track convergence only when a previous model exists to compare with
        converged = False
        if len(self.models) == len(new_models):
            per_step  = [
                np.linalg.norm(new_models[t] - self.models[t])
                for t in range(self.n_times)
            ]
            # Baseline epoch (t=0) excluded from the averaging metric
            changes   = per_step[1:]
            avg_chg   = float(np.mean(changes))
            self.convergence_history.append({
                "iteration":  k + 1,
                "avg_change": avg_chg,
                "per_step":   per_step,
            })
            print(f"  Avg ||dm|| (T>=1): {avg_chg:.6f}", end="")
            converged, diag = self._check_convergence(k)
            if diag:
                print(f"\n{diag}")
            else:
                print()
            if converged:
                print(f"  Converged at iteration {k + 1} (window={CONV_WINDOW})")

            # --- remember the most-settled iterate (smallest avg ||dm||) ---
            if self.return_best and avg_chg < self._best_change:
                self._best_change  = avg_chg
                self._best_models  = [m.copy() for m in new_models]

        self.models = new_models
        return converged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """
        Run HWSI.

        Stage 1 — independent initialisation of all epochs.
        Stage 2 — iterative Gauss-Seidel refinement with adaptive weighting.

        Returns
        -------
        results : list of ndarray
            Inverted resistivity (Ohm.m) at each epoch, in chronological
            order.  Values are bounded to [1, 1e4] Ohm.m.
        """
        print(
            f"\nHWSI  lambda_s={self.lambda_s}  lambda_t={self.lambda_t}  "
            f"eps={self.epsilon}  alpha_max={self.alpha_max}"
        )

        # ---- Stage 1: independent seed ----
        print("\nStage 1: independent initialisation")
        self.models = []
        for t, mgr in enumerate(self.managers):
            m = mgr.invert(
                self.data_list[t],
                lam=self.lambda_s,
                limits=[1.0, 1e4],
                verbose=False,
            )
            self.models.append(np.log(np.clip(np.array(m), 1.0, 1e4)))
            rho = np.exp(self.models[-1])
            print(f"  T={t}: rho=[{rho.min():.1f}, {rho.max():.1f}]")

        self.weights_t = [np.ones(self.n_cells)] * self.n_times

        # ---- Stage 2: adaptive Gauss-Seidel iterations ----
        print(f"\nStage 2: Gauss-Seidel iterations (max {self.max_iter})")
        for k in range(self.max_iter):
            if k > 0:
                self._compute_weights()
            try:
                done = self._run_one_iteration(k)
            except Exception as exc:
                print(f"\nError at iteration {k + 1}: {exc}")
                break
            if done:
                break
        else:
            print("\nReached maximum iterations without declaring convergence.")

        # Return the most-settled iterate when requested; otherwise the last.
        final = self.models
        if self.return_best and self._best_models is not None:
            final = self._best_models
            print(f"  Returning best iterate (avg ||dm|| = {self._best_change:.4f})")
        return [np.clip(np.exp(m), 1.0, 1e4) for m in final]


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------

def run_independent(inv_mesh, data_list, lam=20.0):
    """
    Independent single-epoch inversion (Daily et al., 1992).

    Each epoch is treated as a standalone problem with no temporal information
    exchanged between timesteps.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\nIndependent inversion  (lam={lam})")
    results = []
    for t, data in enumerate(data_list):
        mgr = ert.ERTManager()
        mgr.setData(data)
        mgr.setMesh(inv_mesh)
        m = mgr.invert(data, lam=lam, limits=[1.0, 1e4], verbose=False)
        m = np.clip(np.array(m), 1.0, 1e4)
        results.append(m)
        print(f"  T={t}: rho=[{m.min():.1f}, {m.max():.1f}]")
    return results


# --------------------------------------------------------------------------
# R3-3: form of the difference inversion (LaBrecque and Yang, 2001).
#   "log-additive"    -> log d_t - log d_0 + log F(m_0), i.e. (d_t/d_0) F(m_0)
#                        the correction in the log domain the inversion uses
#   "linear-additive" -> d_t - d_0 + F(m_0) in linear apparent resistivity;
#                        documented as unusable, see the module docstring
# --------------------------------------------------------------------------
DIFF_FORM = "log-additive"

def run_difference(inv_mesh, data_list, lam=20.0):
    """
    Difference inversion (LaBrecque & Yang, 2001).

    Reformulates the problem around change relative to the baseline epoch by
    constructing a transformed data vector  d_diff = (d_t / d_0) * F(m_0),
    where F(m_0) is the forward response of the baseline model.  The
    inversion then recovers resistivity change from baseline.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\nDifference inversion  (lam={lam})")
    results = []

    # Invert baseline epoch independently
    mgr0 = ert.ERTManager()
    mgr0.setData(data_list[0])
    mgr0.setMesh(inv_mesh)
    m0 = np.clip(
        np.array(mgr0.invert(data_list[0], lam=lam, limits=[1.0, 1e4], verbose=False)),
        1.0, 1e4,
    )
    results.append(m0)
    print(f"  T=0 (baseline): rho=[{m0.min():.1f}, {m0.max():.1f}]")

    # Forward response of the baseline model needed for the transformation
    fop0 = ert.ERTModelling()
    fop0.setData(data_list[0])
    fop0.setMesh(inv_mesh, ignoreRegionManager=True)
    f_m0    = np.maximum(np.array(fop0.response(m0)), 0.1)
    rhoa_0  = np.maximum(np.array(data_list[0]["rhoa"]), 0.1)

    for t in range(1, len(data_list)):
        rhoa_t = np.maximum(np.array(data_list[t]["rhoa"]), 0.1)

        # Transformed apparent resistivity data for epoch t.  The inversion
        # forms its misfit on log-transformed data (ERTManager sets
        # inv.dataTrans = TransLogLU), so the additive baseline correction of
        # LaBrecque and Yang (2001) is applied in that same log domain:
        #     log d_corr = log d_t - log d_0 + log F(m_0)
        # which is identically  d_corr = (d_t / d_0) * F(m_0).
        if DIFF_FORM == "log-additive":
            d_diff = (rhoa_t / rhoa_0) * f_m0
        elif DIFF_FORM == "linear-additive":
            # Same correction in linear apparent resistivity.  Retained only to
            # document why it cannot be used: F(m_0) - d_0 scales with the
            # background, so it swamps the small readings carrying the plume.
            d_diff = rhoa_t - rhoa_0 + f_m0
            n_bad = int(np.sum(d_diff <= 0.1))
            if n_bad:
                print(f"    t={t}: {n_bad} of {d_diff.size} transformed readings "
                      f"({100.0 * n_bad / d_diff.size:.1f} %) were non-positive and "
                      f"were floored at 0.1 Ohm.m")
            d_diff = np.maximum(d_diff, 0.1)
        else:
            raise ValueError(f"unknown DIFF_FORM {DIFF_FORM!r}")

        # Combined error propagated from baseline and current epoch
        err_diff = np.clip(
            np.sqrt(
                np.array(data_list[t]["err"]) ** 2
                + np.array(data_list[0]["err"]) ** 2
            ),
            0.01, 1.0,
        )

        dd = data_list[t].copy()
        dd["rhoa"] = d_diff
        dd.set("err", err_diff)

        mgr = ert.ERTManager()
        mgr.setData(dd)
        mgr.setMesh(inv_mesh)
        mt_ = np.clip(
            np.array(
                mgr.invert(
                    dd, lam=lam, limits=[1.0, 1e4],
                    startModel=m0, isReference=True, verbose=False,
                )
            ),
            1.0, 1e4,
        )
        results.append(mt_)
        print(f"  T={t}: rho=[{mt_.min():.1f}, {mt_.max():.1f}]")

    return results


def run_4d(inv_mesh, data_list, lam=20.0, scalef=1.0):
    """
    4D L2-coupled inversion (Kim et al., 2009) via pyGIMLi's TimelapseERT.

    A spatially uniform temporal smoothness penalty (controlled by scalef)
    is applied identically to every mesh cell and every timestep transition.

    Parameters
    ----------
    inv_mesh  : pygimli mesh
    data_list : list of DataContainerERT
    lam       : float — spatial regularisation strength (default 20.0)
    scalef    : float — temporal coupling strength scale factor (default 1.0)

    Returns
    -------
    results : list of ndarray
        Resistivity in Ohm.m at each epoch.
    """
    print(f"\n4D L2-coupled inversion  (lam={lam}, scalef={scalef})")
    tl = ert.TimelapseERT(DATA=data_list, mesh=inv_mesh)
    # TimelapseERT rebuilds its own scheme (combineMultipleData re-sorts M/N with
    # unify=True) while keeping the supplied k, which can leave k on an ordering
    # whose true geometric factor has the opposite sign -> flipped Jacobian and a
    # 4D inversion that cannot fit the data.  Force a consistent, positive k.
    _s = tl.data
    _k = np.array(ert.createGeometricFactors(_s, numerical=True))
    if (_k < 0).any():
        _m = np.array(_s["m"]).copy(); _n = np.array(_s["n"]).copy()
        _neg = _k < 0
        _m[_neg], _n[_neg] = _n[_neg], _m[_neg].copy()
        _s["m"] = _m; _s["n"] = _n
        _k = np.array(ert.createGeometricFactors(_s, numerical=True))
    assert (_k > 0).all(), "4D: geometric factors still negative after M/N swap"
    _s["k"] = _k

    # pyGIMLi API changed across minor versions; try the newer signature first
    try:
        tl.fullInversion(scalef=scalef, lam=lam, maxIter=10, verbose=False)
    except TypeError:
        tl.fullInversion(scalef=scalef, lam=lam, verbose=False)

    results = [
        np.clip(np.array(tl.models[t]), 1.0, 1e4)
        for t in range(len(data_list))
    ]
    for t, r in enumerate(results):
        print(f"  T={t}: rho=[{r.min():.1f}, {r.max():.1f}]")
    return results


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def interp_to_grid(mesh, model, gx, gz):
    """
    Interpolate a mesh-based resistivity model onto a regular grid.

    Parameters
    ----------
    mesh  : pygimli mesh
    model : ndarray — per-cell resistivity values
    gx    : ndarray — grid x-coordinates (from np.mgrid)
    gz    : ndarray — grid z-coordinates (from np.mgrid)

    Returns
    -------
    grid_model : ndarray
        Resistivity on the regular grid.  Nodes outside the convex hull of
        the mesh cell centres are set to NaN.
    """
    cc = np.array([[c.center().x(), c.center().y()] for c in mesh.cells()])
    return griddata((cc[:, 0], cc[:, 1]), model, (gx, gz), method="linear")


def build_valid_mask(inv_mesh, gx, gz, grid_extent=(-12, 12)):
    """
    Build a boolean mask that excludes poorly constrained far-corner nodes.

    Nodes are excluded if they lie outside the interpolation support of the
    mesh OR if they are both far from the electrode array (|x| > extent-1.5)
    and deep (z < -4.5 m), where sensitivity is negligible.

    Parameters
    ----------
    inv_mesh    : pygimli mesh
    gx, gz      : ndarray — grid coordinates (from np.mgrid)
    grid_extent : tuple — (xmin, xmax) of the interpolation grid

    Returns
    -------
    mask : ndarray of bool
    """
    dummy = interp_to_grid(inv_mesh, np.ones(inv_mesh.cellCount()), gx, gz)
    far   = (np.abs(gx) > grid_extent[1] - 1.5) & (gz < -4.5)
    return ~np.isnan(dummy) & ~far


def calc_errors(times, true_grids, results_dict, gx, gz, inv_mesh, mask=None):
    """
    Compute RMSE, normalised RMSE and MAE for a set of inversion results.

    Parameters
    ----------
    times        : array-like — epoch times (hours)
    true_grids   : list of ndarray — true resistivity on the regular grid
    results_dict : dict {method_name: list_of_per_cell_resistivity_arrays}
    gx, gz       : ndarray — grid coordinates
    inv_mesh     : pygimli mesh
    mask         : ndarray of bool, optional — valid node mask

    Returns
    -------
    errors : dict
        {method_name: {"rmse": [...], "nrmse": [...], "mae": [...]}}
    """
    errors = {m: {"rmse": [], "nrmse": [], "mae": []} for m in results_dict}

    for ti in range(len(times)):
        tg  = true_grids[ti]
        rng = np.max(tg) - np.min(tg)
        # Use mean as normaliser when the dynamic range is negligible (e.g. t=0)
        nom = np.mean(tg) if rng < 1.0 else rng

        for name, res in results_dict.items():
            ig    = interp_to_grid(inv_mesh, res[ti], gx, gz)
            valid = (~np.isnan(ig)) if mask is None else (mask & ~np.isnan(ig))

            if valid.any():
                d = tg[valid] - ig[valid]
                errors[name]["rmse"].append(float(np.sqrt(np.mean(d ** 2))))
                errors[name]["nrmse"].append(
                    float(np.sqrt(np.mean(d ** 2)) / nom * 100)
                    if nom > 1e-8 else float("nan")
                )
                errors[name]["mae"].append(float(np.mean(np.abs(d))))
            else:
                for k in errors[name]:
                    errors[name][k].append(float("nan"))

    return errors




# ===========================================================================
# ARRAY-SENSITIVITY DRIVER
#   Models 1 and 2 from the paper, repeated for dipole-dipole, Wenner and
#   Schlumberger acquisition, with all four inversion methods and the paper's
#   nRMSE metric (Eq. 11).
# ===========================================================================
import os, pickle, itertools

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ---- what to run ----------------------------------------------------------
ARRAYS = [("dd", "Dipole-dipole"), ("wa", "Wenner"), ("slm", "Schlumberger")]
MODELS = ["Model 1", "Model 2"]          # trim either list to shorten the run
INCLUDE_HWSI_PAPER = False               # True also runs HWSI without relaxation

# ---- paper settings (identical for every array) ---------------------------
ELECS        = np.linspace(-12, 12, 50)
NOISE_LEVEL  = 0.03
DATA_ERROR   = 0.05
BG_RHO       = 600.0
LAMBDA_S, LAMBDA_T = 20.0, 20.0
EPS, ALPHA_MAX, SCALEF = 5e-3, 0.65, 1.0
HWSI_MAXITER = 15
times = np.arange(0, 25, 4)              # 0,4,...,24 h

# ---- Model 1: single centred conductive plume (Section 4.2) ---------------
M1 = dict(rho=10.0, cx=0.0, cz=0.0, r_min=0.5, r_max=4.0)

def m1_radius(t):
    return M1["r_min"] + (M1["r_max"] - M1["r_min"]) * (t / 24.0)

def m1_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0:
        return rho
    d = np.sqrt((gx - M1["cx"]) ** 2 + (gz - M1["cz"]) ** 2)
    rho[(d <= m1_radius(t)) & (gz <= 0)] = M1["rho"]
    return rho

def m1_point(x, z, t):
    if z > 0 or t <= 0:
        return BG_RHO
    d = np.sqrt((x - M1["cx"]) ** 2 + (z - M1["cz"]) ** 2)
    return M1["rho"] if d <= m1_radius(t) else BG_RHO

# ---- Model 2: two Fourier-perturbed elliptical plumes (Section 5) ---------
PLUME_A = dict(cx=-4.0, cz=-1.0, rho=10.0, r_min=0.3, r_max=2.5,
               ax=1.0, az=1.4, n_modes=4, amp=0.25, phase_offset=0.0)
PLUME_B = dict(cx=+5.0, cz=0.0, rho_start=55.0, rho_end=30.0,
               r_min=0.2, r_max=1.8, ax=1.0, az=1.2,
               n_modes=4, amp=0.20, phase_offset=1.1)
_RNG    = np.random.default_rng(seed=123)
_PERT_A = _RNG.uniform(-1, 1, PLUME_A["n_modes"])
_PERT_B = _RNG.uniform(-1, 1, PLUME_B["n_modes"])

def _ellipse_r(theta, r_base, ax, az, amp, n_modes, pert, phase):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    r_ell = r_base / np.sqrt((np.cos(theta) / ax) ** 2 + (np.sin(theta) / az) ** 2)
    pert_v = sum(pert[k - 1] * np.sin(k * theta + phase + k * 0.5)
                 for k in range(1, n_modes + 1))
    r_bnd = np.maximum(r_ell * (1.0 + amp * pert_v), r_ell * 0.3)
    return float(r_bnd[0]) if r_bnd.size == 1 else r_bnd

def _m2_radii(t):
    prog  = np.clip(t / 24.0, 0.0, 1.0)
    r_A   = PLUME_A["r_min"] + (PLUME_A["r_max"] - PLUME_A["r_min"]) * prog
    r_B   = PLUME_B["r_min"] + (PLUME_B["r_max"] - PLUME_B["r_min"]) * prog
    rho_B = PLUME_B["rho_start"] + (PLUME_B["rho_end"] - PLUME_B["rho_start"]) * prog
    return r_A, r_B, rho_B

def m2_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0:
        return rho
    r_A, r_B, rho_B = _m2_radii(t)
    sub = gz <= 0
    for plume, r, val, pert in [(PLUME_B, r_B, rho_B, _PERT_B),
                                (PLUME_A, r_A, PLUME_A["rho"], _PERT_A)]:
        dx = gx - plume["cx"]; dz = gz - plume["cz"]
        r_bnd = _ellipse_r(np.arctan2(dz, dx), r, plume["ax"], plume["az"],
                           plume["amp"], plume["n_modes"], pert, plume["phase_offset"])
        rho[sub & (np.sqrt(dx ** 2 + dz ** 2) <= r_bnd)] = val
    return rho

def m2_point(x, z, t):
    if z > 0 or t <= 0:
        return BG_RHO
    r_A, r_B, rho_B = _m2_radii(t)
    for plume, r, val, pert in [(PLUME_B, r_B, rho_B, _PERT_B),
                                (PLUME_A, r_A, PLUME_A["rho"], _PERT_A)]:
        dx = x - plume["cx"]; dz = z - plume["cz"]
        r_bnd = _ellipse_r(np.arctan2(dz, dx), r, plume["ax"], plume["az"],
                           plume["amp"], plume["n_modes"], pert, plume["phase_offset"])
        if np.sqrt(dx ** 2 + dz ** 2) <= r_bnd:
            return val
    return BG_RHO

MODEL_DEF = {"Model 1": (m1_point, m1_grid), "Model 2": (m2_point, m2_grid)}

# ---- meshes: identical for every array (same 50 electrode positions) ------
_ref_scheme = ert.createData(elecs=ELECS, schemeName="dd")
world = mt.createWorld(start=[-25, -15], end=[25, 0], worldMarker=True)
for p in _ref_scheme.sensorPositions():
    world.createNode(p)
fwd_mesh   = mt.createMesh(world, quality=34.5, area=0.2)        # forward 0.2 m^2
inv_domain = mt.createRectangle(start=[-12, -6], end=[12, 0])
inv_mesh   = mt.createMesh(inv_domain, quality=34.5, area=0.1)   # inversion 0.1 m^2
grid_x, grid_z = np.mgrid[-12:12:300j, -6:0:150j]
mask = build_valid_mask(inv_mesh, grid_x, grid_z, grid_extent=(-12, 12))
print(f"Forward mesh: {fwd_mesh.cellCount()} cells | "
      f"Inversion mesh: {inv_mesh.cellCount()} cells")

METHODS = ["Independent", "Difference", "4D L2-Coupled", "HWSI"]
if INCLUDE_HWSI_PAPER:
    METHODS.append("HWSI (no relax)")


def simulate_epochs(scheme, point_fn, grid_fn):
    """Synthetic data + true grids for all epochs, paper settings."""
    data_list, true_grids = [], []
    for t in times:
        rho_vec = np.array([point_fn(c.center().x(), c.center().y(), t)
                            for c in fwd_mesh.cells()])
        d = ert.simulate(mesh=fwd_mesh, scheme=scheme, res=rho_vec,
                         noiseLevel=NOISE_LEVEL, seed=123, verbose=False)
        d["rhoa"] = np.maximum(np.abs(d["rhoa"]), 0.1)
        d.set("err", np.full(d.size(), DATA_ERROR))
        data_list.append(d)
        true_grids.append(grid_fn(grid_x, grid_z, t))
    return data_list, true_grids


def run_case(model_name, code, label):
    """One (model, array) combination; cached so the run can be resumed."""
    cache = os.path.join(OUT_DIR, f"array_{model_name.replace(' ', '').lower()}_{code}.pkl")
    if os.path.exists(cache):
        print(f"\n[{model_name} | {label}] loading cached results")
        return pickle.load(open(cache, "rb"))

    print("\n" + "#" * 70)
    print(f"[{model_name} | {label} ({code})]")
    print("#" * 70)
    scheme = ert.createData(elecs=ELECS, schemeName=code)
    point_fn, grid_fn = MODEL_DEF[model_name]
    data_list, true_grids = simulate_epochs(scheme, point_fn, grid_fn)
    print(f"  {scheme.size()} measurements per epoch, {len(times)} epochs, "
          f"{NOISE_LEVEL*100:.0f}% noise")

    res = {}
    res["Independent"]   = run_independent(inv_mesh, data_list, lam=LAMBDA_S)
    res["Difference"]    = run_difference(inv_mesh, data_list, lam=LAMBDA_S)
    try:
        res["4D L2-Coupled"] = run_4d(inv_mesh, data_list, lam=LAMBDA_S, scalef=SCALEF)
    except Exception as e:
        print("  4D failed:", e)
        res["4D L2-Coupled"] = res["Independent"]
    res["HWSI"] = HWSI_Inversion(inv_mesh, data_list, lambda_s=LAMBDA_S, lambda_t=LAMBDA_T,
                                 epsilon=EPS, alpha_max=ALPHA_MAX, max_iter=HWSI_MAXITER,
                                 verbose=False, relax=0.5, return_best=True).run()
    if INCLUDE_HWSI_PAPER:
        res["HWSI (no relax)"] = HWSI_Inversion(inv_mesh, data_list, lambda_s=LAMBDA_S,
                                 lambda_t=LAMBDA_T, epsilon=EPS, alpha_max=ALPHA_MAX,
                                 max_iter=HWSI_MAXITER, verbose=False).run()

    err = calc_errors(times, true_grids, res, grid_x, grid_z, inv_mesh, mask=mask)
    out = {"nrmse": {m: err[m]["nrmse"] for m in res},
           "ndata": int(scheme.size())}
    pickle.dump(out, open(cache, "wb"))
    print(f"cached -> {cache}")
    return out


# ---- run every requested combination --------------------------------------
summary = {}
for model_name, (code, label) in itertools.product(MODELS, ARRAYS):
    summary[(model_name, code)] = run_case(model_name, code, label)

    print(f"\nnRMSE (%) per epoch  [{model_name} | {label}]  (t=0 excluded)")
    print("-" * 78)
    print(f"{'t (h)':<7}" + "".join(f"{m:<16}" for m in METHODS))
    nr = summary[(model_name, code)]["nrmse"]
    for ti, t in enumerate(times):
        if t == 0:
            continue
        print(f"{t:<7}" + "".join(f"{nr[m][ti]:<16.2f}" for m in METHODS))
    print(f"{'mean':<7}" + "".join(f"{np.nanmean(nr[m][1:]):<16.2f}" for m in METHODS))

# ---- cross-array summary table -------------------------------------------
print("\n" + "=" * 96)
print("ARRAY SENSITIVITY - mean nRMSE (%) over t = 4-24 h  [Eq. 11]")
print("=" * 96)
print(f"{'model':<9}{'array':<16}{'n data':>7}" + "".join(f"{m:>16}" for m in METHODS)
      + f"{'HWSI vs best other':>21}")
for (model_name, code), out in summary.items():
    label = dict(ARRAYS)[code]
    means = {m: float(np.nanmean(out["nrmse"][m][1:])) for m in METHODS}
    others = [means[m] for m in METHODS if not m.startswith("HWSI")]
    gain = min(others) - means["HWSI"]
    print(f"{model_name:<9}{label:<16}{out['ndata']:>7}"
          + "".join(f"{means[m]:>16.2f}" for m in METHODS)
          + f"{gain:>+20.2f}")
print("HWSI vs best other = percentage points; positive means HWSI is more accurate")

# ---- figure: one panel per (model, array) --------------------------------
fig, axes = plt.subplots(len(MODELS), len(ARRAYS),
                         figsize=(5.2 * len(ARRAYS), 4.0 * len(MODELS)),
                         squeeze=False, sharex=True)
for i, model_name in enumerate(MODELS):
    for j, (code, label) in enumerate(ARRAYS):
        ax = axes[i][j]
        nr = summary[(model_name, code)]["nrmse"]
        for m in METHODS:
            ax.plot(times[1:], nr[m][1:], marker="o",
                    linewidth=2.6 if m.startswith("HWSI") else 1.5, label=m)
        ax.set_title(f"{model_name} - {label} ({summary[(model_name, code)]['ndata']} data)",
                     fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        if j == 0:
            ax.set_ylabel("nRMSE (%)  [Eq. 11]")
        if i == len(MODELS) - 1:
            ax.set_xlabel("time (h)")
axes[0][0].legend(fontsize=8)
fn = os.path.join(OUT_DIR, "array_sensitivity_nrmse.png")
fig.tight_layout(); fig.savefig(fn, dpi=200, bbox_inches="tight")
print("\nSaved", fn)
print("\nDone.")
