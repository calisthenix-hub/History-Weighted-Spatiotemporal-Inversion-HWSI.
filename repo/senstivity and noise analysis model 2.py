"""Model 2 sensitivity: two sweeps across all four methods.

  lambda_s over decades {0.1, 1, 10, 100, 1000} at 3 per cent noise and a
  5 per cent floor (R3), and the noise level over {1, 3, 5, 7} per cent with
  the error floor matched to it (R2-9, R3), at lambda_s = lambda_t = 20.

Accuracy is the paper's Eq. 11 nRMSE, with log10 RMSE alongside it.  This is
the Model 2 sensitivity the submitted paper did not carry.  DO_LAMBDA_S and
DO_NOISE pick the sweeps; each noise level caches to noise_level_<n>.pkl, so an
interrupted run resumes.

FAST trades the paper mesh for a coarse one.  It gives the right trends in
about half an hour but the numbers do not line up with the paper's tables, so
leave it off for anything quotable.

Reproducing the published lambda_s column.  The table and figure in the
response letter were produced with HWSI in its plain form - max_iter 15, no
under-relaxation, last iterate - giving 123.82, 69.66, 52.68, 59.89 and 51.50
per cent across the five decades.  This file runs the stabilised form instead
(relax 0.5, best iterate), which is what the array study and the field
experiment use, so its HWSI column can differ by a point or two.  Independent,
Difference and 4D L2-coupled are unaffected and reproduce exactly.  Set
HWSI_RELAX = 1.0 and HWSI_BEST = False below to reproduce the published
numbers.

Self-contained: the engine is bundled so the file runs from any folder.
"""

import os
import numpy as np
import time
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
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
    errors = {m: {"rmse": [], "nrmse": [], "mae": [], "logrms": []} for m in results_dict}

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
                # log-resistivity RMSE (decades): background-fair metric (R3-8)
                lt_ = np.log10(np.clip(tg[valid], 1e-3, None))
                li_ = np.log10(np.clip(ig[valid], 1e-3, None))
                errors[name]["logrms"].append(float(np.sqrt(np.mean((lt_ - li_) ** 2))))
            else:
                for k in errors[name]:
                    errors[name][k].append(float("nan"))

    return errors



# --- the two sweeps, on the paper's Section 4 Model 2 setup ------------

# which sweeps to run
DO_LAMBDA_S = False   # set True to repeat the lambda_s sweep (hours: it has
                      # no per-value cache, unlike the noise levels below)
DO_NOISE    = True

# FAST=True: coarse mesh + capped/relaxed HWSI -> a sweep finishes in ~20-35 min
#            and still gives valid sensitivity TRENDS (relative comparison).
# FAST=False: the fine paper mesh + full HWSI -> ACCURATE but HOURS per sweep.
FAST = False          # paper mesh + full HWSI (hours, but consistent with Table 6)
if FAST:
    FWD_AREA, INV_AREA = 0.6, 0.5
    HWSI_MAXITER, HWSI_RELAX, HWSI_BEST = 6, 0.5, True
else:
    FWD_AREA, INV_AREA = 0.2, 0.1        # paper mesh
    # Stabilised HWSI, as used for the array study and the field experiment.
    # For the published lambda_s column use 15, 1.0, False - see the header.
    HWSI_MAXITER, HWSI_RELAX, HWSI_BEST = 15, 0.5, True

# ---- paper Section 4.1 / 4.2 (Model 2) exact setup ----
scheme = ert.createData(elecs=np.linspace(-12, 12, 50), schemeName="dd")
world  = mt.createWorld(start=[-25, -15], end=[25, 0], worldMarker=True)
for p in scheme.sensorPositions():
    world.createNode(p)
fwd_mesh   = mt.createMesh(world, quality=34.5, area=FWD_AREA)
inv_domain = mt.createRectangle(start=[-12, -6], end=[12, 0])
inv_mesh   = mt.createMesh(inv_domain, quality=34.5, area=INV_AREA)
grid_x, grid_z = np.mgrid[-12:12:300j, -6:0:150j]
print(f"[{'FAST' if FAST else 'PAPER-FINE'}] "
      f"inv mesh area={INV_AREA}, HWSI max_iter={HWSI_MAXITER}, relax={HWSI_RELAX}")
print(f"Forward mesh: {fwd_mesh.cellCount()} cells | "
      f"Inversion mesh: {inv_mesh.cellCount()} cells")

BG_RHO  = 600.0
PLUME_A = dict(cx=-4.0, cz=-1.0, rho=10.0, r_min=0.3, r_max=2.5,
               ax=1.0, az=1.4, n_modes=4, amp=0.25, phase_offset=0.0)
PLUME_B = dict(cx=5.0, cz=0.0, rho_start=55.0, rho_end=30.0, r_min=0.2, r_max=1.8,
               ax=1.0, az=1.2, n_modes=4, amp=0.20, phase_offset=1.1)
_RNG    = np.random.default_rng(123)
_PERT_A = _RNG.uniform(-1, 1, PLUME_A["n_modes"])
_PERT_B = _RNG.uniform(-1, 1, PLUME_B["n_modes"])


def _ellipse_r(theta, r_base, ax, az, amp, n_modes, pert, phase):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    r_ell = r_base / np.sqrt((np.cos(theta) / ax) ** 2 + (np.sin(theta) / az) ** 2)
    pert_v = sum(pert[k - 1] * np.sin(k * theta + phase + k * 0.5)
                 for k in range(1, n_modes + 1))
    r_bnd = np.maximum(r_ell * (1.0 + amp * pert_v), r_ell * 0.3)
    return float(r_bnd[0]) if r_bnd.size == 1 else r_bnd


def _current_radii(t):
    prog = np.clip(t / 24.0, 0.0, 1.0)
    rA = PLUME_A["r_min"] + (PLUME_A["r_max"] - PLUME_A["r_min"]) * prog
    rB = PLUME_B["r_min"] + (PLUME_B["r_max"] - PLUME_B["r_min"]) * prog
    rhoB = PLUME_B["rho_start"] + (PLUME_B["rho_end"] - PLUME_B["rho_start"]) * prog
    return rA, rB, rhoB


def get_true_resistivity(x, z, t):
    if z > 0 or t <= 0:
        return BG_RHO
    rA, rB, rhoB = _current_radii(t)
    for plume, r, val, pert in [(PLUME_B, rB, rhoB, _PERT_B),
                                (PLUME_A, rA, PLUME_A["rho"], _PERT_A)]:
        dx, dz = x - plume["cx"], z - plume["cz"]
        r_bnd = _ellipse_r(np.arctan2(dz, dx), r, plume["ax"], plume["az"],
                           plume["amp"], plume["n_modes"], pert, plume["phase_offset"])
        if np.sqrt(dx ** 2 + dz ** 2) <= r_bnd:
            return val
    return BG_RHO


def get_true_resistivity_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0:
        return rho
    rA, rB, rhoB = _current_radii(t)
    sub = gz <= 0
    for plume, r, val, pert in [(PLUME_B, rB, rhoB, _PERT_B),
                                (PLUME_A, rA, PLUME_A["rho"], _PERT_A)]:
        dx, dz = gx - plume["cx"], gz - plume["cz"]
        r_bnd = _ellipse_r(np.arctan2(dz, dx), r, plume["ax"], plume["az"],
                           plume["amp"], plume["n_modes"], pert, plume["phase_offset"])
        rho[sub & (np.sqrt(dx ** 2 + dz ** 2) <= r_bnd)] = val
    return rho


times = np.arange(0, 25, 4)   # 0,4,...,24 h (paper 4.1)
_TRUE_GRIDS = [get_true_resistivity_grid(grid_x, grid_z, t) for t in times]
mask = build_valid_mask(inv_mesh, grid_x, grid_z, grid_extent=(-12, 12))
METHODS = ["Independent", "Difference", "4D L2-Coupled", "HWSI"]


def simulate_data(noise, floor):
    """The seven epochs at one noise level and error floor."""
    dl = []
    for t in times:
        rv = np.array([get_true_resistivity(c.center().x(), c.center().y(), t)
                       for c in fwd_mesh.cells()])
        d = ert.simulate(mesh=fwd_mesh, scheme=scheme, res=rv,
                         noiseLevel=noise, seed=123, verbose=False)
        d["rhoa"] = np.maximum(np.abs(d["rhoa"]), 0.1)
        d.set("err", np.full(d.size(), floor))
        dl.append(d)
    return dl


def run_all_methods(data_list, ls, lt=20.0, tag=None):
    """All four methods on one dataset.  Returns mean nRMSE and mean log10
    RMSE per method, and warns if a solution is pinned at the resistivity
    bounds, which means it should not be believed.
    """
    fns = {
        "Independent":   lambda: run_independent(inv_mesh, data_list, lam=ls),
        "Difference":    lambda: run_difference(inv_mesh, data_list, lam=ls),
        "4D L2-Coupled": lambda: run_4d(inv_mesh, data_list, lam=ls, scalef=1.0),
        "HWSI":          lambda: HWSI_Inversion(
            inv_mesh, data_list, lambda_s=ls, lambda_t=lt,
            epsilon=5e-3, alpha_max=0.65, max_iter=HWSI_MAXITER, verbose=False,
            relax=HWSI_RELAX, return_best=HWSI_BEST).run(),
    }
    out, outlog = {}, {}
    for m, fn in fns.items():
        try:
            res = fn()
            err = calc_errors(times, _TRUE_GRIDS, {m: res},
                              grid_x, grid_z, inv_mesh, mask=mask)
            out[m]    = float(np.nanmean(err[m]["nrmse"][1:]))
            outlog[m] = float(np.nanmean(err[m]["logrms"][1:]))
            # health check: fraction of cells pinned at the [1, 1e4] bounds
            pin = np.mean([np.mean((np.asarray(r) <= 1.01) | (np.asarray(r) >= 9.99e3))
                           for r in res])
            if pin > 0.001:
                print(f"    {m:<15} WARNING: {pin*100:.2f} % of cells pinned at the "
                      f"resistivity bounds - solution is not trustworthy")
            if tag:                       # cache models so metrics never need a re-run
                import pickle, os
                with open(f"models_{tag}_{m.replace(' ', '_')}.pkl", "wb") as fh:
                    pickle.dump([np.asarray(r) for r in res], fh)
            print(f"    {m:<15} nRMSE = {out[m]:6.2f}%   log-RMSE = {outlog[m]:5.3f}")
        except Exception as e:
            out[m] = outlog[m] = float("nan")
            print(f"    {m:<15} FAILED: {e}")
    return out, outlog


def _print_table(title, xlabel, xvals, rows, fmt="{:<15.2f}"):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)
    print(f"{xlabel:<12}" + "".join(f"{m:<15}" for m in METHODS))
    for i, xv in enumerate(xvals):
        print(f"{xv:<12g}" + "".join(fmt.format(rows[m][i]) for m in METHODS))
    best = [min(METHODS, key=lambda m: (np.inf if np.isnan(rows[m][i]) else rows[m][i]))
            for i in range(len(xvals))]
    print("-" * 66)
    print("best method: " + ", ".join(f"{xv:g}->{best[i]}"
                                       for i, xv in enumerate(xvals)))


def _save_curve(xvals, rows, xlabel, title, fname, logx=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        ax.plot(xvals, rows[m], marker="o",
                linewidth=2.6 if m == "HWSI" else 1.7, label=m)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean nRMSE (%)   [Eq. 11]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both", linestyle="--")
    ax.legend()
    try:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"Saved {fname}")
    except OSError:
        alt = fname.replace(".png", f"_{int(time.time())}.png")
        fig.savefig(alt, dpi=200, bbox_inches="tight")
        print(f"Saved {alt} (original was locked)")
    plt.close(fig)


# --- sweep 1: lambda_s, at 3 per cent noise and a 5 per cent floor -----
if DO_LAMBDA_S:
    LAMBDA_S_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]     # decades (Reviewer 3)
    print("\n########  SWEEP 1: lambda_s  (noise 3% / floor 5%)  ########")
    data_ref = simulate_data(0.03, 0.05)
    ls_rows = {m: [] for m in METHODS}
    ls_log  = {m: [] for m in METHODS}
    for ls in LAMBDA_S_GRID:
        print(f"\n----  lambda_s = {ls:g}  ----")
        out, outlog = run_all_methods(data_ref, ls, lt=20.0, tag=f"ls{ls:g}")
        for m in METHODS:
            ls_rows[m].append(out[m]); ls_log[m].append(outlog[m])
    _print_table("Model 2 — mean nRMSE (%) vs spatial regularisation lambda_s",
                 "lambda_s", LAMBDA_S_GRID, ls_rows)
    _print_table("Model 2 \u2014 mean log10-resistivity RMSE (decades) vs lambda_s",
                 "lambda_s", LAMBDA_S_GRID, ls_log, fmt="{:<15.3f}")
    _save_curve(LAMBDA_S_GRID, ls_rows, r"spatial regularisation $\lambda_s$",
                "Model 2 — sensitivity to spatial regularisation",
                "model2_sensitivity_lambda_s.png", logx=True)


# --- sweep 2: noise, with the floor matched to it ----------------------
if DO_NOISE:
    NOISE_GRID = [0.01, 0.03, 0.05, 0.07]               # matched noise = floor
    print("\n########  SWEEP 2: noise (matched floor), lambda_s=20  ########")
    nz_rows = {m: [] for m in METHODS}
    nz_log  = {m: [] for m in METHODS}
    for nz in NOISE_GRID:
        print(f"\n----  noise = floor = {nz*100:g}%  ----")
        lvl_cache = f"noise_level_{nz*100:g}.pkl"
        if os.path.exists(lvl_cache):
            import pickle as _pk
            out, outlog = _pk.load(open(lvl_cache, "rb"))
            print("    (loaded from cache)")
            for m in METHODS:
                print(f"    {m:<15} nRMSE = {out[m]:6.2f}%   log-RMSE = {outlog[m]:5.3f}")
            for m in METHODS:
                nz_rows[m].append(out[m]); nz_log[m].append(outlog[m])
            continue
        dl = simulate_data(nz, nz)                      # matched noise & floor
        out, outlog = run_all_methods(dl, ls=20.0, lt=20.0, tag=f"nz{nz*100:g}")
        import pickle as _pk
        _pk.dump((out, outlog), open(lvl_cache, "wb"))
        for m in METHODS:
            nz_rows[m].append(out[m]); nz_log[m].append(outlog[m])
    _print_table("Model 2 — mean nRMSE (%) vs matched noise level",
                 "noise", [n * 100 for n in NOISE_GRID], nz_rows)
    _save_curve([n * 100 for n in NOISE_GRID], nz_rows, "matched noise level (%)",
                "Model 2 — sensitivity to noise (matched floor)",
                "model2_sensitivity_noise.png", logx=False)

print("\nAll requested sweeps complete.")
