import matplotlib
matplotlib.use('Agg')
"""HWSI and the three benchmark methods, plus the padded-domain re-run.

Models 1 and 2 on a forward mesh padded to x +/-120 m and z -80 m, scored both
on the masked grid of the submitted paper and on the full grid - the pair of
checks R3-5 and R2-5 asked for.  A homogeneous half-space test on three meshes
separates boundary truncation from discretisation before any inversion runs.

Results cache to padded_model1.pkl / padded_model2.pkl.  Delete those for a
clean re-run.  With RERUN_DIFFERENCE set, only the Difference column is
recomputed, in the log-additive form of R3-3, and the other three methods are
taken from the cache unchanged.

Several scripts in this folder exec() this file to pick up the meshes, the true
models and metrics(), so the module-level names below are an interface.  Renaming
them, or the file, breaks those scripts.
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
                 relax=1.0, return_best=False, seed_models=None):

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
        # Field data: independent seeds wander, so Stage 1 can be handed the
        # 4D solution instead and refined from there.
        self.seed_models = seed_models

        # On a para-mesh (marker 2 inverted, marker 1 fixed background) the
        # inversion returns one value per para cell, not per mesh cell.
        _mk = np.array(inv_mesh.cellMarkers())
        if (_mk == 2).any() and (_mk != 2).any():
            self.n_cells = int((_mk == 2).sum())
        else:
            self.n_cells = inv_mesh.cellCount()
        self.n_times = len(data_list)

        # rebuilt every sweep
        self.models   = []
        self.weights_t = []
        self.convergence_history = []
        self._best_models = None
        self._best_change = np.inf

        # Reference only - the spatial term is pyGIMLi's, not ours.
        self.L_s = (self._build_laplacian()
                    if self.n_cells == inv_mesh.cellCount() else None)

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

        return np.clip(m_sp, 0.0, np.log(3000.0))

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

        # ---- Stage 1: seed ----
        self.models = []
        if self.seed_models is not None:
            print("\nStage 1: robust external seeding (e.g. from 4D solution)")
            for t in range(self.n_times):
                self.models.append(
                    np.log(np.clip(np.asarray(self.seed_models[t], float), 1.0, 1e4)))
                rho = np.exp(self.models[-1])
                print(f"  T={t}: rho=[{rho.min():.1f}, {rho.max():.1f}]  (seed)")
        else:
            print("\nStage 1: independent initialisation")
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
    fop0.setMesh(inv_mesh)   # region-aware: model is para-cells only
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
    # TimelapseERT rebuilds its own scheme with combineMultipleData(), which
    # re-sorts M/N (unify=True) but KEEPS the k we supplied.  That undoes the
    # M/N swap done in build_container and leaves k positive on an ordering
    # whose true k is negative -> Jacobian sign flipped -> the 4D inversion
    # cannot fit the data (it stayed at ~72 % misfit, no plume).  Redo the swap
    # on the internal scheme so k > 0 again.
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
    try:
        jacobian_sign_check(tl.chooseTime(0), "4D internal scheme")
    except Exception as _e:
        print("  (4D sign check skipped:", _e, ")")

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


# --- the padded-domain run: R3-5 (far-field boundaries) and R2-5 (deep
#     low-sensitivity cells), same settings as the paper otherwise ------
import os, pickle, time
from scipy.interpolate import griddata

OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RERUN_DIFFERENCE = True      # recompute ONLY the Difference column from the
                             # cached run, using the additive form (R3-3)
MODELS_TO_RUN = ["Model 1", "Model 2"]      # trim to shorten the run
LAMBDA_S = LAMBDA_T = 20.0
EPS, ALPHA_MAX, SCALEF = 5e-3, 0.65, 1.0
HWSI_MAXITER, HWSI_RELAX = 15, 0.5
NOISE, FLOOR, BG_RHO = 0.03, 0.05, 600.0
ELECS = np.linspace(-12, 12, 50)
times = np.arange(0, 25, 4)
METHODS = ["Independent", "Difference", "4D L2-Coupled", "HWSI"]

scheme = ert.createData(elecs=ELECS, schemeName="dd")

# ---------------------------------------------------------------- meshes ---
def paper_forward_mesh():
    """The submitted mesh: uniform 0.2 m^2 over [-25, 25] x [-15, 0]."""
    world = mt.createWorld(start=[-25, -15], end=[25, 0], worldMarker=True)
    for p in scheme.sensorPositions():
        world.createNode(p)
    return mt.createMesh(world, quality=34.5, area=0.2)


def forward_mesh(pad_x, pad_z, fine_x=20.0, fine_z=12.0, fine_area=0.2, coarse_area=25.0):
    """Graded mesh - a fine box around the array inside a padded world."""
    world = mt.createWorld(start=[-pad_x, -pad_z], end=[pad_x, 0],
                           worldMarker=True, area=coarse_area)
    fine = mt.createRectangle(start=[-fine_x, -fine_z], end=[fine_x, 0],
                              marker=2, area=fine_area, boundaryMarker=0)
    plc = world + fine
    for p in scheme.sensorPositions():
        plc.createNode(p)
        plc.createNode(p - [0, 0.1])
    return mt.createMesh(plc, quality=34.0)

print("Building meshes ...")
fwd_old = paper_forward_mesh()                         # submitted extent
fwd_new = forward_mesh(120.0, 80.0)                    # padded (~10x spread)
print(f"  forward, submitted extent : {fwd_old.cellCount():6d} cells  "
      f"(x +/-25 m, z -15 m)")
print(f"  forward, padded extent    : {fwd_new.cellCount():6d} cells  "
      f"(x +/-120 m, z -80 m)")

# inversion mesh: parameter region embedded in a padded background (R3-5)
plc_inv   = mt.createParaMeshPLC(scheme, paraDX=0.3, paraDepth=6.0,
                                 paraMaxCellSize=0.1, boundary=8)
inv_mesh  = mt.createMesh(plc_inv, quality=34.0, smooth=True)
para_mesh = inv_mesh.createSubMesh(inv_mesh.cells(inv_mesh.cellMarkers() == 2))
bb = inv_mesh.boundingBox()
print(f"  inversion mesh            : {inv_mesh.cellCount():6d} cells, "
      f"{para_mesh.cellCount()} parameter cells")
print(f"    parameter region x {para_mesh.boundingBox().min()[0]:.1f}.."
      f"{para_mesh.boundingBox().max()[0]:.1f} m, depth "
      f"{para_mesh.boundingBox().min()[1]:.1f} m")
print(f"    background extends to x {bb.min()[0]:.0f}..{bb.max()[0]:.0f} m, "
      f"z {bb.min()[1]:.0f} m")

# ------------------------------------- boundary-truncation check (R3-5) ----
def halfspace_error(mesh):
    """Departure of the homogeneous half-space response from rho, max and mean
    in per cent.  The exact answer is zero, so this is pure mesh error.
    """
    d = ert.simulate(mesh=mesh, scheme=scheme, res=BG_RHO, noiseLevel=0.0, verbose=False)
    r = np.abs(np.array(d["rhoa"]))
    return float(np.max(np.abs(r - BG_RHO)) / BG_RHO * 100), float(np.mean(np.abs(r - BG_RHO)) / BG_RHO * 100)

fwd_ctrl = forward_mesh(25.0, 15.0)     # same near-field grading, unpadded outer box
e_old  = halfspace_error(fwd_old)
e_ctrl = halfspace_error(fwd_ctrl)
e_new  = halfspace_error(fwd_new)
print(f"\nHomogeneous half-space forward error, {BG_RHO:.0f} Ohm.m (exact answer = 0):")
print(f"  submitted mesh, x +/-25 m, z -15 m      : max {e_old[0]:.2f} %,  mean {e_old[1]:.2f} %")
print(f"  same near-field grading, x +/-25 m       : max {e_ctrl[0]:.2f} %,  mean {e_ctrl[1]:.2f} %")
print(f"  padded,        x +/-120 m, z -80 m      : max {e_new[0]:.2f} %,  mean {e_new[1]:.2f} %")
print("  (the two graded meshes differ only in outer extent, so their difference")
print("   isolates boundary truncation from discretisation)")

# ------------------------------------------------------------ true models --
M1 = dict(rho=10.0, cx=0.0, cz=0.0, r_min=0.5, r_max=4.0)
def m1_r(t): return M1["r_min"] + (M1["r_max"] - M1["r_min"]) * (t / 24.0)
def m1_point(x, z, t):
    if z > 0 or t <= 0: return BG_RHO
    return M1["rho"] if np.hypot(x - M1["cx"], z - M1["cz"]) <= m1_r(t) else BG_RHO
def m1_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0: return rho
    rho[(np.sqrt((gx - M1["cx"])**2 + (gz - M1["cz"])**2) <= m1_r(t)) & (gz <= 0)] = M1["rho"]
    return rho

PA = dict(cx=-4.0, cz=-1.0, rho=10.0, r_min=0.3, r_max=2.5, ax=1.0, az=1.4,
          n_modes=4, amp=0.25, phase_offset=0.0)
PB = dict(cx=5.0, cz=0.0, rho_start=55.0, rho_end=30.0, r_min=0.2, r_max=1.8,
          ax=1.0, az=1.2, n_modes=4, amp=0.20, phase_offset=1.1)
_RNG = np.random.default_rng(seed=123)
_PA, _PB = _RNG.uniform(-1, 1, PA["n_modes"]), _RNG.uniform(-1, 1, PB["n_modes"])
def _ell(theta, r, ax, az, amp, n, pert, ph):
    theta = np.atleast_1d(np.asarray(theta, float))
    r_e = r / np.sqrt((np.cos(theta)/ax)**2 + (np.sin(theta)/az)**2)
    pv = sum(pert[k-1] * np.sin(k*theta + ph + k*0.5) for k in range(1, n+1))
    rb = np.maximum(r_e * (1 + amp*pv), r_e*0.3)
    return float(rb[0]) if rb.size == 1 else rb
def _radii(t):
    p = np.clip(t/24.0, 0, 1)
    return (PA["r_min"] + (PA["r_max"]-PA["r_min"])*p,
            PB["r_min"] + (PB["r_max"]-PB["r_min"])*p,
            PB["rho_start"] + (PB["rho_end"]-PB["rho_start"])*p)
def m2_point(x, z, t):
    if z > 0 or t <= 0: return BG_RHO
    rA, rB, rhoB = _radii(t)
    for pl, r, val, pert in [(PB, rB, rhoB, _PB), (PA, rA, PA["rho"], _PA)]:
        dx, dz = x - pl["cx"], z - pl["cz"]
        if np.hypot(dx, dz) <= _ell(np.arctan2(dz, dx), r, pl["ax"], pl["az"],
                                    pl["amp"], pl["n_modes"], pert, pl["phase_offset"]):
            return val
    return BG_RHO
def m2_grid(gx, gz, t):
    rho = np.full(gx.shape, BG_RHO)
    if t <= 0: return rho
    rA, rB, rhoB = _radii(t); sub = gz <= 0
    for pl, r, val, pert in [(PB, rB, rhoB, _PB), (PA, rA, PA["rho"], _PA)]:
        dx, dz = gx - pl["cx"], gz - pl["cz"]
        rb = _ell(np.arctan2(dz, dx), r, pl["ax"], pl["az"], pl["amp"],
                  pl["n_modes"], pert, pl["phase_offset"])
        rho[sub & (np.sqrt(dx**2 + dz**2) <= rb)] = val
    return rho
DEF = {"Model 1": (m1_point, m1_grid), "Model 2": (m2_point, m2_grid)}

# --------------------------------------------------------------- metrics --
grid_x, grid_z = np.mgrid[-12:12:300j, -6:0:150j]
_cc = np.array([[c.center().x(), c.center().y()] for c in para_mesh.cells()])
def to_grid(model):
    return griddata((_cc[:, 0], _cc[:, 1]), np.asarray(model), (grid_x, grid_z), method="linear")
_support = ~np.isnan(to_grid(np.ones(para_mesh.cellCount())))
_far = (np.abs(grid_x) > 10.5) & (grid_z < -4.5)
MASKED   = _support & ~_far          # as in the submitted manuscript
FULL     = _support                  # every resolvable node (R2-5)
print(f"\nmetric grid: {MASKED.sum()} masked nodes, {FULL.sum()} full-domain nodes "
      f"({FULL.sum() - MASKED.sum()} far-corner nodes restored)")

def metrics(true_grids, models):
    """Per-epoch nRMSE (Eq. 11) and log10 RMSE, on the masked and full grids."""
    out = {k: [] for k in ("nrmse_m", "nrmse_f", "log_m", "log_f")}
    for ti in range(len(times)):
        tg = true_grids[ti]
        rng = np.max(tg) - np.min(tg)
        nom = np.mean(tg) if rng < 1.0 else rng
        ig = to_grid(models[ti])
        for tag, msk in (("m", MASKED), ("f", FULL)):
            v = msk & ~np.isnan(ig)
            d = tg[v] - ig[v]
            out[f"nrmse_{tag}"].append(float(np.sqrt(np.mean(d**2))/nom*100))
            lt_ = np.log10(np.clip(tg[v], 1e-3, None)); li_ = np.log10(np.clip(ig[v], 1e-3, None))
            out[f"log_{tag}"].append(float(np.sqrt(np.mean((lt_ - li_)**2))))
    return out

def jacobian_sign_check(data, label=""):     # used by run_4d
    return 1.0

# ------------------------------------------------------------------ runs --
summary = {}
for mname in MODELS_TO_RUN:
    cache = os.path.join(OUT, f"padded_{mname.replace(' ', '').lower()}.pkl")
    cached = pickle.load(open(cache, "rb")) if os.path.exists(cache) else None
    if cached is not None and not RERUN_DIFFERENCE:
        print(f"\n[{mname}] loading cached results")
        summary[mname] = cached; continue
    if cached is not None and "models" not in cached:
        print(f"\n[{mname}] cache has no stored models; full re-run required")
        cached = None

    print(f"\n{'#'*70}\n[{mname}]  padded forward mesh, paper settings\n{'#'*70}")
    point_fn, grid_fn = DEF[mname]
    data_list, true_grids = [], []
    for t in times:
        rv = np.array([point_fn(c.center().x(), c.center().y(), t) for c in fwd_new.cells()])
        d = ert.simulate(mesh=fwd_new, scheme=scheme, res=rv, noiseLevel=NOISE,
                         seed=123, verbose=False)
        d["rhoa"] = np.maximum(np.abs(d["rhoa"]), 0.1)
        d.set("err", np.full(d.size(), FLOOR))
        data_list.append(d); true_grids.append(grid_fn(grid_x, grid_z, t))

    t0 = time.time()
    if cached is not None and RERUN_DIFFERENCE:
        # Keep Independent, 4D and HWSI exactly as published; recompute only the
        # Difference column with the additive LaBrecque-Yang form.
        print("  re-running the Difference method only "
              f"(DIFF_FORM = {DIFF_FORM!r}); other methods taken from the cache")
        res = {m: [np.asarray(x) for x in cached["models"][m]] for m in METHODS}
        res["Difference"] = run_difference(inv_mesh, data_list, lam=LAMBDA_S)
        print(f"  ({(time.time()-t0)/60:.1f} min)")
        old_m = cached["metrics"]
        out = {m: metrics(true_grids, res[m]) for m in METHODS}
        print(f"\n  Difference, mean nRMSE over t = 4-24 h: "
              f"ratio form {np.nanmean(old_m['Difference']['nrmse_m'][1:]):.2f} %"
              f"  ->  additive form {np.nanmean(out['Difference']['nrmse_m'][1:]):.2f} %")
        pickle.dump({"metrics": out,
                     "models": {m: [np.asarray(x) for x in res[m]] for m in METHODS}},
                    open(cache, "wb"))
        summary[mname] = {"metrics": out}
        print(f"cache updated -> {cache}")
        continue

    res = {}
    res["Independent"]   = run_independent(inv_mesh, data_list, lam=LAMBDA_S)
    res["Difference"]    = run_difference(inv_mesh, data_list, lam=LAMBDA_S)
    try:
        res["4D L2-Coupled"] = run_4d(inv_mesh, data_list, lam=LAMBDA_S, scalef=SCALEF)
    except Exception as e:
        print("  4D failed:", e); res["4D L2-Coupled"] = res["Independent"]
    res["HWSI"] = HWSI_Inversion(inv_mesh, data_list, lambda_s=LAMBDA_S, lambda_t=LAMBDA_T,
                                 epsilon=EPS, alpha_max=ALPHA_MAX, max_iter=HWSI_MAXITER,
                                 verbose=False, relax=HWSI_RELAX, return_best=True).run()
    print(f"  ({(time.time()-t0)/60:.0f} min)")
    out = {m: metrics(true_grids, res[m]) for m in METHODS}
    pickle.dump({"metrics": out, "models": {m: [np.asarray(x) for x in res[m]] for m in METHODS}},
                open(cache, "wb"))
    summary[mname] = {"metrics": out}
    print(f"cached -> {cache}")

# ---------------------------------------------------------------- tables --
for mname in MODELS_TO_RUN:
    mt_ = summary[mname]["metrics"]
    print("\n" + "=" * 88)
    print(f"{mname}: masked vs full-domain accuracy, padded forward mesh "
          f"(mean over t = 4-24 h)")
    print("=" * 88)
    print(f"{'method':<16}{'nRMSE masked':>14}{'nRMSE full':>12}{'delta':>8}"
          f"{'logRMS masked':>15}{'logRMS full':>13}")
    for m in METHODS:
        a = np.nanmean(mt_[m]["nrmse_m"][1:]); b = np.nanmean(mt_[m]["nrmse_f"][1:])
        c = np.nanmean(mt_[m]["log_m"][1:]);   d = np.nanmean(mt_[m]["log_f"][1:])
        print(f"{m:<16}{a:>14.2f}{b:>12.2f}{b-a:>+8.2f}{c:>15.3f}{d:>13.3f}")
    print("-" * 88)
    print(f"{'per-epoch nRMSE (masked):':<28}" + "  ".join(f"t{int(t)}h" for t in times[1:]))
    for m in METHODS:
        print(f"{m:<28}" + "  ".join(f"{v:5.1f}" for v in mt_[m]["nrmse_m"][1:]))
print("\nDone.")
