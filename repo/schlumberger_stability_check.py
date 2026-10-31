"""Model 1 with a Schlumberger array: did HWSI fail to settle, or is the
array the limitation?

In the array-sensitivity run this was the one case where HWSI did not settle.
The update norm bounced around (12.9, 3.0, 5.3, 2.1) and nRMSE grew with time,
where every other case decayed below 0.1.  Rather than argue from one number,
this repeats that single case with stronger under-relaxation and more
iterations and records nRMSE after every sweep, so the curve answers it.

Everything else - meshes, noise realisation, regularisation - is the same as
the array run, so only relax and max_iter differ.

Writes slm_model1_stability.pkl and slm_model1_stability.png.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert


# Convergence: the mean update norm has to flatten off over a window of
# sweeps rather than drop below a tolerance on a single one, because the
# weight field needs a few sweeps to settle before the models do.

REL_CONV_TOL = 0.03    # maximum relative improvement still accepted as plateau
CV_TOL       = 0.15    # maximum coefficient of variation within the window
CONV_WINDOW  = 3       # number of iterations in the plateau check window
MIN_ITERS    = 4       # earliest iteration at which convergence can be declared



class HWSI_Inversion:
    """History-weighted spatiotemporal inversion.

    Every cell gets its own temporal blending coefficient.  The weight for a
    cell is the reciprocal of how far it moved between the last two accepted
    models, normalised by the median over the mesh and clipped at the 95th
    percentile; alpha follows as lambda_t*w / (lambda_s + lambda_t*w), capped
    at alpha_max.  Weights are rebuilt at the start of every sweep, so the
    coupling field settles together with the models rather than being fixed
    in advance.

    Defaults are the paper values.  relax < 1 under-relaxes the sweep, which
    damps the oscillation between the data fit and the temporal prior;
    return_best keeps the iterate with the smallest mean update instead of
    whichever one the loop happened to end on.
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
        # relax=1.0 and return_best=False give the original behaviour.
        self.relax       = float(relax)
        self.return_best = bool(return_best)

        self.n_cells = inv_mesh.cellCount()
        self.n_times = len(data_list)

        # rebuilt every sweep
        self.models   = []
        self.weights_t = []
        self.convergence_history = []
        self._best_models = None
        self._best_change = np.inf

        # Reference only - the spatial term is pyGIMLi's, not ours.
        self.L_s = self._build_laplacian()

        # one manager per epoch, reused across sweeps
        self.managers = []
        for d in data_list:
            mgr = ert.ERTManager()
            mgr.setData(d)
            mgr.setMesh(inv_mesh)
            self.managers.append(mgr)


    def _build_laplacian(self):
        """Graph Laplacian of the cell adjacency.  Kept for reference only -
        pyGIMLi does the spatial regularisation itself.
        """
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
        """Per-cell weights from the two most recent models.  Below t = 2 there
        is no history to measure, so the weight falls back to 1.
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
        """One epoch: spatial inversion, then blend with epoch t-1 as it stands
        in the current sweep.  Works in log resistivity throughout.
        """
        # warm start from the previous sweep where there is one
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

        # the baseline is never blended, and neither is an epoch whose
        # predecessor has not been solved yet in this sweep
        if t > 0 and len(current_pass) >= t:
            lw    = self.lambda_t * self.weights_t[t]
            alpha = np.clip(lw / (self.lambda_s + lw), 0.0, self.alpha_max)
            m_sp  = (1.0 - alpha) * m_sp + alpha * current_pass[t - 1]

        return np.clip(m_sp, 0.0, np.log(1e4))

    def _check_convergence(self, k):
        """Plateau test on the last CONV_WINDOW sweeps.  Returns the verdict and
        a diagnostic line, empty until the window is full.
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
        """One Gauss-Seidel sweep over all epochs.  True if the plateau test passes."""
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

        # under-relax against the previous iterate; the baseline is left out
        if self.relax < 1.0 and len(self.models) == len(new_models):
            for t in range(1, self.n_times):
                new_models[t] = ((1.0 - self.relax) * self.models[t]
                                 + self.relax * new_models[t])

        # nothing to compare against on the first sweep
        converged = False
        if len(self.models) == len(new_models):
            per_step  = [
                np.linalg.norm(new_models[t] - self.models[t])
                for t in range(self.n_times)
            ]
            # baseline excluded
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

            # keep the quietest sweep seen so far
            if self.return_best and avg_chg < self._best_change:
                self._best_change  = avg_chg
                self._best_models  = [m.copy() for m in new_models]

        self.models = new_models
        return converged


    def run(self):
        """Stage 1 seeds every epoch, stage 2 refines them.  Returns resistivity
        per epoch in chronological order.
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
        self.iter_models = []          # snapshot per iteration (stability study)
        for k in range(self.max_iter):
            if k > 0:
                self._compute_weights()
            try:
                done = self._run_one_iteration(k)
            except Exception as exc:
                print(f"\nError at iteration {k + 1}: {exc}")
                break
            self.iter_models.append([np.array(m, copy=True) for m in self.models])
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


# --- the three benchmark methods ---------------------------------------

def run_independent(inv_mesh, data_list, lam=20.0):
    """Every epoch on its own.  The no-coupling baseline (Daily et al., 1992)."""
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


# R3-3.  The inversion forms its misfit on log data, so the LaBrecque-Yang
# correction belongs in the log domain: log d_t - log d_0 + log F(m_0),
# which is (d_t/d_0)*F(m_0).  The linear form is kept below only to show
# why it cannot be used.
DIFF_FORM = "log-additive"

def run_difference(inv_mesh, data_list, lam=20.0):
    """Difference inversion (LaBrecque and Yang, 2001).

    Inverts the transformed data (d_t / d_0) * F(m_0) with the baseline model
    as the reference, so what comes out is change from baseline rather than
    absolute resistivity.
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
    """4D L2-coupled inversion (Kim et al., 2009) through pyGIMLi's
    TimelapseERT.  One scalar temporal weight for every cell and every
    transition, which is the thing HWSI replaces.
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


# --- gridding and error metrics ----------------------------------------

def interp_to_grid(mesh, model, gx, gz):
    """Cell values onto the regular grid.  Nodes outside the hull of the cell
    centres come back NaN.
    """
    cc = np.array([[c.center().x(), c.center().y()] for c in mesh.cells()])
    return griddata((cc[:, 0], cc[:, 1]), model, (gx, gz), method="linear")


def build_valid_mask(inv_mesh, gx, gz, grid_extent=(-12, 12)):
    """Grid nodes worth scoring: inside the interpolation hull, and not both
    far from the line and deep, where there is no sensitivity to speak of.
    """
    dummy = interp_to_grid(inv_mesh, np.ones(inv_mesh.cellCount()), gx, gz)
    far   = (np.abs(gx) > grid_extent[1] - 1.5) & (gz < -4.5)
    return ~np.isnan(dummy) & ~far


def calc_errors(times, true_grids, results_dict, gx, gz, inv_mesh, mask=None):
    """Per-epoch RMSE, nRMSE and MAE for each method, on the masked grid."""
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




# --- the run -----------------------------------------------------------
import os, pickle

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

RELAX     = 0.3      # was 0.5
MAX_ITER  = 25       # was 15
ARRAY     = "slm"
ELECS     = np.linspace(-12, 12, 50)
NOISE_LEVEL, DATA_ERROR = 0.03, 0.05
BG_RHO    = 600.0
LAMBDA_S, LAMBDA_T = 20.0, 20.0
EPS, ALPHA_MAX = 5e-3, 0.65
times = np.arange(0, 25, 4)

# reference numbers from the array-sensitivity run (mean nRMSE, t = 4-24 h)
REF = {"Independent": 77.63, "Difference": 70.12, "4D L2-Coupled": 102.34,
       "HWSI (relax 0.5, 15 it)": 78.10}

# ---- Model 1 truth (Section 4.2) -----------------------------------------
M1 = dict(rho=10.0, cx=0.0, cz=0.0, r_min=0.5, r_max=4.0)

def m1_radius(t):
    return M1["r_min"] + (M1["r_max"] - M1["r_min"]) * (t / 24.0)

def m1_point(x, z, t):
    if z > 0 or t <= 0:
        return BG_RHO
    d = np.sqrt((x - M1["cx"]) ** 2 + (z - M1["cz"]) ** 2)
    return M1["rho"] if d <= m1_radius(t) else BG_RHO

def m1_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0:
        return rho
    d = np.sqrt((gx - M1["cx"]) ** 2 + (gz - M1["cz"]) ** 2)
    rho[(d <= m1_radius(t)) & (gz <= 0)] = M1["rho"]
    return rho

# ---- same meshes and data as the array run (same seed) -------------------
_ref = ert.createData(elecs=ELECS, schemeName="dd")
world = mt.createWorld(start=[-25, -15], end=[25, 0], worldMarker=True)
for p in _ref.sensorPositions():
    world.createNode(p)
fwd_mesh   = mt.createMesh(world, quality=34.5, area=0.2)
inv_domain = mt.createRectangle(start=[-12, -6], end=[12, 0])
inv_mesh   = mt.createMesh(inv_domain, quality=34.5, area=0.1)
grid_x, grid_z = np.mgrid[-12:12:300j, -6:0:150j]
mask = build_valid_mask(inv_mesh, grid_x, grid_z, grid_extent=(-12, 12))
print(f"Forward mesh: {fwd_mesh.cellCount()} cells | Inversion mesh: {inv_mesh.cellCount()} cells")

scheme = ert.createData(elecs=ELECS, schemeName=ARRAY)
print(f"Model 1 | Schlumberger: {scheme.size()} measurements per epoch")
data_list, true_grids = [], []
for t in times:
    rho_vec = np.array([m1_point(c.center().x(), c.center().y(), t) for c in fwd_mesh.cells()])
    d = ert.simulate(mesh=fwd_mesh, scheme=scheme, res=rho_vec,
                     noiseLevel=NOISE_LEVEL, seed=123, verbose=False)
    d["rhoa"] = np.maximum(np.abs(d["rhoa"]), 0.1)
    d.set("err", np.full(d.size(), DATA_ERROR))
    data_list.append(d)
    true_grids.append(m1_grid(grid_x, grid_z, t))

def mean_nrmse(models):
    """Mean nRMSE over t = 4-24 h, and the per-epoch values."""
    e = calc_errors(times, true_grids, {"x": models}, grid_x, grid_z, inv_mesh, mask=mask)
    return float(np.nanmean(e["x"]["nrmse"][1:])), e["x"]["nrmse"]

# ---- the run --------------------------------------------------------------
print(f"\nHWSI stability run: relax={RELAX}, max_iter={MAX_ITER}")
hw = HWSI_Inversion(inv_mesh, data_list, lambda_s=LAMBDA_S, lambda_t=LAMBDA_T,
                    epsilon=EPS, alpha_max=ALPHA_MAX, max_iter=MAX_ITER,
                    verbose=False, relax=RELAX, return_best=True)
res_best = hw.run()                      # best iterate (what the tables use)

# ---- nRMSE after every iteration -----------------------------------------
curve = []
for k, snap in enumerate(getattr(hw, "iter_models", []), start=1):
    models = [np.clip(np.exp(m), 1.0, 1e4) for m in snap]
    mval, _ = mean_nrmse(models)
    curve.append(mval)
    print(f"  iteration {k:2d}: mean nRMSE = {mval:6.2f}%")

best_mean, best_per_epoch = mean_nrmse(res_best)
last_mean = curve[-1] if curve else float("nan")

print("\n" + "=" * 78)
print("Model 1 | Schlumberger - HWSI stability check")
print("=" * 78)
print(f"{'t (h)':<8}{'nRMSE (%)':>12}")
for ti, t in enumerate(times):
    if t == 0:
        continue
    print(f"{t:<8}{best_per_epoch[ti]:>12.2f}")
print("-" * 78)
for k, v in REF.items():
    print(f"{k:<28}{v:>8.2f}")
print(f"{'HWSI (relax 0.3, ' + str(MAX_ITER) + ' it)':<28}{best_mean:>8.2f}   "
      f"(last iterate {last_mean:.2f})")
print("-" * 78)
if curve:
    print(f"nRMSE over iterations: first {curve[0]:.2f}, min {min(curve):.2f} "
          f"(iteration {int(np.argmin(curve)) + 1}), last {curve[-1]:.2f}")
    tail = curve[-5:] if len(curve) >= 5 else curve
    spread = (max(tail) - min(tail)) / max(np.mean(tail), 1e-9) * 100
    print(f"spread over the last {len(tail)} iterations: {spread:.1f}% of the mean "
          f"(<5% = settled, >15% = still oscillating)")
    print(f"vs the previous HWSI setting: {REF['HWSI (relax 0.5, 15 it)'] - best_mean:+.2f} "
          f"points;  vs the best other method: "
          f"{min(REF['Independent'], REF['Difference'], REF['4D L2-Coupled']) - best_mean:+.2f} points")

pickle.dump({"curve": curve, "best_mean": best_mean, "per_epoch": best_per_epoch,
             "relax": RELAX, "max_iter": MAX_ITER, "ref": REF},
            open(os.path.join(OUT_DIR, "slm_model1_stability.pkl"), "wb"))

# ---- figure ---------------------------------------------------------------
if curve:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(range(1, len(curve) + 1), curve, marker="o", lw=2.2,
            label=f"HWSI (relax {RELAX}, {MAX_ITER} it)")
    for k, v in REF.items():
        ax.axhline(v, ls="--", lw=1.2, alpha=0.7)
        ax.text(len(curve) * 0.55, v, k, fontsize=8, va="bottom")
    ax.set_xlabel("Gauss-Seidel iteration")
    ax.set_ylabel("mean nRMSE (%), t = 4-24 h")
    ax.set_title("Model 1 | Schlumberger - HWSI convergence")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9)
    fn = os.path.join(OUT_DIR, "slm_model1_stability.png")
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print("\nSaved", fn)
print("\nDone.")
