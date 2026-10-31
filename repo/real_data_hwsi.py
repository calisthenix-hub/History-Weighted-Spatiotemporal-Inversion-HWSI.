import matplotlib
matplotlib.use('Agg')
"""Field validation of HWSI on the saline-injection time-lapse data set.

Seven epochs at four-hourly intervals over 24 h, 24 electrodes at 0.5 m,
dipole-dipole, Res2DInv general-array .Dat.  215 quadrupoles are common to all
epochs; QC keeps |k| < 1500 and rhoa in [5, 2500], leaving 139.  About 350 g of
NaCl in 40 L went in between electrodes 11 and 12 (x = 5.0-5.5 m) after t = 4 h,
so the target is roughly 10 Ohm.m against a 190 Ohm.m background.

The mesh puts the electrodes on mesh nodes.  A plain rectangle without them
gave up to 30 per cent forward error on a homogeneous half-space, which the
inversions then happily fitted as structure.

Two settings are run: the paper's (lambda_s = lambda_t = 20, 5 per cent floor)
and a field-realistic one (80 / 30, 8 per cent floor).  HWSI is run twice, once
from independent seeds and once seeded from the 4D solution, because on noisy
field data the independent seeds wander.

Geometric factors are the thing to be careful about.  With the instrument's
A,B,M,N ordering pyGIMLi returns k < 0 for every reading; taking |k| leaves the
forward response positive but flips the sign of the Jacobian, which images the
saline body as resistive.  M and N are swapped instead wherever k < 0, in the
container and again inside TimelapseERT, which rebuilds its own scheme and
would otherwise undo the swap.  A finite-difference check on the Jacobian runs
before anything is inverted and stops the script if it disagrees.

Writes real_sections_<tag>.png, real_change_<tag>.png, real_data_v8_<tag>.pkl
and the tables printed at the end.
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
            start = np.clip(np.exp(self.models[t]), 3.0, 3000.0)

        m_sp = self.managers[t].invert(
            self.data_list[t],
            lam=self.lambda_s,
            limits=[3.0, 3000.0],
            startModel=start,
            verbose=False,
        )
        m_sp = np.log(np.clip(np.array(m_sp), 3.0, 3000.0))

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
                    np.log(np.clip(np.asarray(self.seed_models[t], float), 3.0, 3000.0)))
                rho = np.exp(self.models[-1])
                print(f"  T={t}: rho=[{rho.min():.1f}, {rho.max():.1f}]  (seed)")
        else:
            print("\nStage 1: independent initialisation")
            for t, mgr in enumerate(self.managers):
                m = mgr.invert(
                    self.data_list[t],
                    lam=self.lambda_s,
                    limits=[3.0, 3000.0],
                    verbose=False,
                )
                self.models.append(np.log(np.clip(np.array(m), 3.0, 3000.0)))
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
        return [np.clip(np.exp(m), 3.0, 3000.0) for m in final]


# --- the three benchmark methods ---------------------------------------

def run_independent(inv_mesh, data_list, lam=20.0):
    """Every epoch on its own.  The no-coupling baseline (Daily et al., 1992)."""
    print(f"\nIndependent inversion  (lam={lam})")
    results = []
    for t, data in enumerate(data_list):
        mgr = ert.ERTManager()
        mgr.setData(data)
        mgr.setMesh(inv_mesh)
        m = mgr.invert(data, lam=lam, limits=[3.0, 3000.0], verbose=False)
        m = np.clip(np.array(m), 3.0, 3000.0)
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
        np.array(mgr0.invert(data_list[0], lam=lam, limits=[3.0, 3000.0], verbose=False)),
        3.0, 3000.0,
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
                    dd, lam=lam, limits=[3.0, 3000.0],
                    startModel=m0, isReference=True, verbose=False,
                )
            ),
            3.0, 3000.0,
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
        np.clip(np.array(tl.models[t]), 3.0, 3000.0)
        for t in range(len(data_list))
    ]
    for t, r in enumerate(results):
        print(f"  T={t}: rho=[{r.min():.1f}, {r.max():.1f}]")
    return results

# --- the field driver ---------------------------------------------------
import os, sys, glob, pickle

EPOCH_FILES = ["2.Dat", "3.Dat", "4.h5new.Dat", "5.Dat", "6.Dat", "7.Dat", "8.Dat"]
TIMES       = [0, 4, 8, 12, 16, 20, 24]          # hours
ERR_FLOOR   = 0.08                               # field-realistic floor (real ERT
                                                 # is noisier than the 5% synthetic)
LAMBDA_S = 80.0                                  # stronger spatial regularisation
LAMBDA_T = 30.0                                  # so field seeds stay stable
FIELD_TUNED = True                               # seed HWSI from the stable 4D solve
EPS, ALPHA_MAX, SCALEF = 5e-3, 0.65, 1.0
METHODS = ["Independent", "Difference", "4D L2-Coupled", "HWSI"]

# ---- locate the folder that actually holds the 7 epoch files --------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_CANDIDATES = [
    _HERE,
    os.path.join(_HERE, "Garveet GPT"),
    os.path.join(_HERE, "field_data"),
    os.environ.get("HWSI_FIELD_DATA", ""),      # set this to the folder of .Dat files
    os.getcwd(),
]
def _find_data_dir():
    for c in _CANDIDATES:
        if c and all(os.path.exists(os.path.join(c, f)) for f in EPOCH_FILES):
            return c
    return None

print("=" * 64)
print("REAL-DATA HWSI FIELD VALIDATION  (single-plume, 7 epochs)")
print("=" * 64)
DATA_DIR = _find_data_dir()
if DATA_DIR is None:
    print("ERROR: could not find the 7 epoch files "
          + ", ".join(EPOCH_FILES) + ".")
    print("Looked in:")
    for c in _CANDIDATES:
        print("   ", c)
    print("\nFIX: set DATA_DIR below to the folder holding those .Dat files, "
          "or\n     run this script from inside the 'Garveet GPT' folder.")
    # DATA_DIR = r"C:\\full\\path\\to\\Garveet GPT"   # <-- uncomment & edit if needed
    sys.exit(1)
print("Using data folder:", DATA_DIR)


# Res2DInv general-array .Dat -> DataContainerERT
def parse_dat(path):
    rows = []
    for ln in open(path, errors="ignore").read().splitlines():
        t = ln.split()
        if len(t) >= 9 and t[0] == "4":
            try:
                v = list(map(float, t[1:]))
            except ValueError:
                continue
            if len(v) < 9:
                continue
            A, B, M, N = (v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7])
            rhoa = v[8]
            if rhoa > 0:
                rows.append((A, B, M, N, rhoa))
    return rows


def cfg_key(r):
    """Quadrupole identity by electrode x-position."""
    return tuple(round(e[0], 3) for e in r[:4])


def common_config_order(all_rows):
    """Quadrupoles present in every epoch, kept in the baseline's order so the
    difference, 4D and HWSI formulations all see the same measurements in the
    same places.
    """
    sets = [set(cfg_key(r) for r in rows) for rows in all_rows]
    common = set.intersection(*sets)
    # keep them in the baseline epoch's order for determinism
    order = [cfg_key(r) for r in all_rows[0] if cfg_key(r) in common]
    return order


def all_electrode_x(all_rows):
    xs = set()
    for rows in all_rows:
        for (A, B, M, N, r) in rows:
            for e in (A, B, M, N):
                xs.add(round(e[0], 3))
    return sorted(xs)


def build_container(rows, elec_x, key_order):
    """Build a DataContainerERT restricted to key_order.

    The geometric factors need care; see the note in the body.
    """
    by_key = {cfg_key(r): r for r in rows}
    rows = [by_key[k] for k in key_order]          # aligned, identical order
    data = pg.DataContainerERT()
    for x in elec_x:
        data.createSensor([float(x), 0.0])
    pos = {x: i for i, x in enumerate(elec_x)}
    data.resize(len(rows))
    a, b, m, n, rhoa = [], [], [], [], []
    for (A, B, M, N, r) in rows:
        a.append(pos[round(A[0], 3)]); b.append(pos[round(B[0], 3)])
        m.append(pos[round(M[0], 3)]); n.append(pos[round(N[0], 3)])
        rhoa.append(r)
    data["a"] = np.array(a); data["b"] = np.array(b)
    data["m"] = np.array(m); data["n"] = np.array(n)
    data["rhoa"] = np.array(rhoa)
    data["valid"] = np.ones(len(rows))
    # Geometric factors -- SIGN MATTERS.  With the instrument's A,B,M,N order
    # pyGIMLi's k is negative for every dipole-dipole reading.  Replacing k by
    # |k| (as the previous version did) leaves the forward response positive
    # (pyGIMLi takes abs) but FLIPS THE SIGN OF THE JACOBIAN, so Gauss-Newton
    # pushed conductive changes resistive.  Fix: swap M<->N wherever k<0, which
    # flips both the potential and k, so k>0 and everything is consistent.
    k = np.array(ert.createGeometricFactors(data, numerical=True))
    if (k < 0).any():
        mm = np.array(data["m"]).copy(); nn = np.array(data["n"]).copy()
        neg = k < 0
        mm[neg], nn[neg] = nn[neg], mm[neg].copy()
        data["m"] = mm; data["n"] = nn
        k = np.array(ert.createGeometricFactors(data, numerical=True))
    assert (k > 0).all(), "geometric factors still negative after M/N swap"
    data["k"] = k
    data["r"] = data["rhoa"] / k
    data.set("err", np.full(len(rows), ERR_FLOOR))   # 5% floor (paper)
    return data


# No ground truth here, so the fit metric is the forward response of the
# recovered model against what was measured.
def data_rms(mesh, data, model_rho):
    """Log-space RMS misfit in per cent between measured and predicted
    apparent resistivity.  The plain relative RMS explodes on the handful of
    very low readings over the plume and is not a fair fit metric here.
    """
    try:
        fop = ert.ERTModelling()
        fop.setData(data)
        fop.setMesh(mesh)          # region-aware: model is para cells
        pred = np.abs(np.array(fop.response(model_rho)))
        obs = np.abs(np.array(data["rhoa"]))
        good = (obs > 0) & (pred > 0) & np.isfinite(pred)
        return float(np.sqrt(np.mean((np.log(obs[good]) - np.log(pred[good])) ** 2)) * 100)
    except Exception as e:
        print("   data_rms failed:", e)
        return float("nan")


def temporal_roughness(models):
    """Mean ||log m_t - log m_{t-1}|| over the monitored epochs."""
    lm = [np.log(np.clip(np.asarray(m), 3.0, 3000.0)) for m in models]
    d = [np.linalg.norm(lm[t] - lm[t - 1]) for t in range(1, len(lm))]
    return float(np.mean(d))


# --- load and QC --------------------------------------------------------
# Ground truth supplied by the experimenter: ~350 g NaCl in 40 L water injected
# between electrodes 11 and 12 (x = 5.0-5.5 m) at t = 4 h; in-ground resistivity
# of the saline body ~10 Ohm.m against a ~190 Ohm.m background.
INJ_X0, INJ_X1 = 5.0, 5.5
ZONE = dict(x0=4.5, x1=6.0, zmin=-0.75)          # scoring window around it

print("Loading real ERT epochs...")
all_rows  = [parse_dat(os.path.join(DATA_DIR, f)) for f in EPOCH_FILES]
elec_x    = all_electrode_x(all_rows)
key_order = common_config_order(all_rows)
print(f"  electrodes: {len(elec_x)} at x = {elec_x[0]}..{elec_x[-1]} m")
print(f"  per-epoch measurements: {[len(r) for r in all_rows]} -> {len(key_order)} common")

K_MAX = 1500.0
RHO_MIN, RHO_MAX = 5.0, 2500.0
def qc_filter(all_rows, elec_x, key_order):
    ref  = build_container(all_rows[0], elec_x, key_order)
    keep = np.abs(np.array(ref["k"])) < K_MAX
    for rows in all_rows:
        by   = {cfg_key(r): r for r in rows}
        rhoa = np.array([by[k][4] for k in key_order])
        keep &= (rhoa >= RHO_MIN) & (rhoa <= RHO_MAX)
    return [k for k, ok in zip(key_order, keep) if ok]
_n0 = len(key_order)
key_order = qc_filter(all_rows, elec_x, key_order)
print(f"  QC (|k|<{K_MAX:.0f}): {_n0} -> {len(key_order)} measurements kept")
base_data = [build_container(r, elec_x, key_order) for r in all_rows]

# Para-mesh with the electrodes as nodes.  A plain rectangle without them
# gives up to 30 per cent forward error, which the inversions then fit.
plc       = mt.createParaMeshPLC(base_data[0], paraDX=0.25, paraDepth=3.0,
                                 paraMaxCellSize=0.03, boundary=4)
inv_mesh  = mt.createMesh(plc, quality=33.5, smooth=True)
para_mesh = inv_mesh.createSubMesh(inv_mesh.cells(inv_mesh.cellMarkers() == 2))
print(f"Inversion mesh: {inv_mesh.cellCount()} cells, {para_mesh.cellCount()} inverted (para) cells")

# Does the Jacobian point the same way as the forward response?  Finite
# differences on a 10 per cent conductive perturbation at the injection.
def jacobian_sign_check(data, label):
    fop = ert.ERTModelling(); fop.setData(data); fop.setMesh(inv_mesh)
    npara = para_mesh.cellCount()
    m0 = np.full(npara, 190.0)
    r0 = np.array(fop.response(pg.Vector(m0)))
    fop.createJacobian(pg.Vector(m0))
    J = pg.utils.gmat2numpy(fop.jacobian())
    cc = np.array([[c.center()[0], c.center()[1]] for c in para_mesh.cells()])
    sel = (np.abs(cc[:, 0] - 0.5 * (INJ_X0 + INJ_X1)) < 0.3) & (cc[:, 1] > -0.3)
    m1 = m0.copy(); m1[sel] *= 0.9
    fd  = np.array(fop.response(pg.Vector(m1))) - r0
    lin = J[:, sel] @ (m1[sel] - m0[sel])
    top = np.argsort(-np.abs(fd))[:15]
    agree = float(np.mean(np.sign(fd[top]) == np.sign(lin[top])))
    print(f"  Jacobian sign check [{label}]: {agree*100:.0f}% of the 15 most "
          f"sensitive readings agree with finite differences")
    return agree

print("\nPre-flight Jacobian check")
_old = pg.DataContainerERT(base_data[0])
_old["m"], _old["n"] = np.array(base_data[0]["n"]), np.array(base_data[0]["m"])
_old["k"] = np.abs(np.array(_old["k"]))             # original order, |k|
jacobian_sign_check(_old, "previous version, |k|")
_ok = jacobian_sign_check(base_data[0], "this version, M/N swapped, k>0")
if _ok < 0.8:
    raise SystemExit("Jacobian disagrees with the forward response -- stop and "
                     "send this output; inversions would be meaningless.")

cc   = np.array([[c.center().x(), c.center().y()] for c in para_mesh.cells()])
zone = (cc[:, 0] >= ZONE["x0"]) & (cc[:, 0] <= ZONE["x1"]) & (cc[:, 1] >= ZONE["zmin"])
bgz  = (((cc[:, 0] >= 0.5) & (cc[:, 0] <= 2.5)) | ((cc[:, 0] >= 9.0) & (cc[:, 0] <= 11.0))) \
       & (cc[:, 1] >= -1.5)

# --- runs: the paper's settings and the field ones ----------------------
RUNS = [   # tag,           lambda_s, lambda_t, error floor
    ("paper_ls20",      20.0,     20.0,     0.05),
    ("field_ls80",      80.0,     30.0,     0.08),
]
HWSI_MAXITER = 10
ONLY_RUN = None     # e.g. "field_ls80" to run just one setting (halves run time)
METHODS = ["Independent", "Difference", "4D L2-Coupled", "HWSI", "HWSI (4D seed)"]

def with_error(dlist, floor):
    out = []
    for d in dlist:
        e = d.copy(); e.set("err", np.full(e.size(), floor)); out.append(e)
    return out

def run_all(tag, ls, lt, floor):
    cache = os.path.join(DATA_DIR, f"real_data_v8_{tag}.pkl")
    if os.path.exists(cache):
        print(f"\n[{tag}] loading cached results"); return pickle.load(open(cache, "rb"))
    print(f"\n{'#'*70}\n[{tag}]  lambda_s={ls}  lambda_t={lt}  error floor={floor*100:.0f}%\n{'#'*70}")
    dl = with_error(base_data, floor)
    res = {}
    # Reuse the v7 run for the methods the 4D fix cannot change (Independent,
    # Difference and plain HWSI never touch TimelapseERT), so only the 4D solve
    # and the 4D-seeded HWSI have to be recomputed.
    old_cache = os.path.join(DATA_DIR, f"real_data_v7_{tag}.pkl")
    reuse = {}
    if os.path.exists(old_cache):
        try:
            reuse = pickle.load(open(old_cache, "rb")).get("res", {})
            print("  reusing Independent / Difference / HWSI from the previous run")
        except Exception as e:
            print("  (could not reuse previous run:", e, ")")
    res["Independent"] = reuse.get("Independent") or run_independent(inv_mesh, dl, lam=ls)
    res["Difference"]  = reuse.get("Difference")  or run_difference(inv_mesh, dl, lam=ls)
    try:
        res["4D L2-Coupled"] = run_4d(inv_mesh, dl, lam=ls, scalef=SCALEF)
    except Exception as e:
        print("4D failed:", e); res["4D L2-Coupled"] = res["Independent"]
    res["HWSI"] = reuse.get("HWSI") or HWSI_Inversion(
                                 inv_mesh, dl, lambda_s=ls, lambda_t=lt, epsilon=EPS,
                                 alpha_max=ALPHA_MAX, max_iter=HWSI_MAXITER, verbose=False,
                                 relax=0.3, return_best=True).run()
    res["HWSI (4D seed)"] = HWSI_Inversion(inv_mesh, dl, lambda_s=ls, lambda_t=lt, epsilon=EPS,
                                 alpha_max=ALPHA_MAX, max_iter=HWSI_MAXITER, verbose=False,
                                 relax=0.3, return_best=True,
                                 seed_models=res["4D L2-Coupled"]).run()
    fit = {m: [data_rms(inv_mesh, dl[t], res[m][t]) for t in range(len(dl))] for m in METHODS}
    out = {"res": res, "fit": fit}
    pickle.dump(out, open(cache, "wb")); print(f"cached -> {cache}")
    return out

# --- scoring against the known injection --------------------------------
def score(res):
    rows = {}
    for m in METHODS:
        z = np.array([np.median(np.asarray(e)[zone]) for e in res[m]])
        b = np.array([np.median(np.asarray(e)[bgz])  for e in res[m]])
        post = z[1:].mean() / z[0]                       # t = 4..24 h vs t = 0
        rows[m] = dict(zone=z, zchg=(post - 1) * 100,
                       bchg=(b[1:].mean() / b[0] - 1) * 100,
                       zmin=float(np.min([np.min(np.asarray(e)[zone]) for e in res[m][1:]])),
                       rough=temporal_roughness(res[m]))
    return rows

def figures(tag, res):
    allv = np.concatenate([np.log10(np.asarray(res[m][t])) for m in METHODS for t in range(7)])
    vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
    for kind in ("sections", "change"):
        fig, axes = plt.subplots(len(METHODS), 7 if kind == "sections" else 6,
                                 figsize=(21, 2.3 * len(METHODS)))
        for i, m in enumerate(METHODS):
            L = [np.log10(np.asarray(e)) for e in res[m]]
            cols = range(7) if kind == "sections" else range(1, 7)
            for j, t in enumerate(cols):
                ax = axes[i, j]
                val = L[t] if kind == "sections" else L[t] - L[0]
                kw = dict(cMap="Spectral", cMin=vmin, cMax=vmax) if kind == "sections" \
                     else dict(cMap="RdBu_r", cMin=-0.5, cMax=0.5)
                pg.show(para_mesh, val, ax=ax, colorBar=False, showMesh=False, **kw)
                ax.plot(elec_x, np.zeros(len(elec_x)), "k.", ms=3)
                ax.plot([INJ_X0, INJ_X1], [0.05, 0.05], "m-", lw=4)
                ax.set_xlim(-0.5, 12.0); ax.set_ylim(-3, 0.15)
                ax.set_xlabel(""); ax.set_ylabel(m if j == 0 else "", fontsize=9)
                if j: ax.set_yticklabels([])
                if i == 0:
                    ax.set_title(f"t={TIMES[t]} h" + ("" if kind == "sections" else " vs t=0"), fontsize=9)
        if kind == "sections":
            sm = plt.cm.ScalarMappable(cmap="Spectral", norm=plt.Normalize(vmin, vmax))
            lab = "log10 resistivity (Ohm.m)   red = conductive"
        else:
            sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-0.5, 0.5))
            lab = "log10(rho_t / rho_0)   blue = more conductive"
        fig.colorbar(sm, ax=axes, shrink=0.6, label=lab)
        fig.suptitle(f"Real data [{tag}] - {kind}.  Magenta bar = injection (electrodes 11-12, t=4 h); "
                     "depth in m on vertical axis", fontsize=11)
        fn = os.path.join(DATA_DIR, f"real_{kind}_{tag}.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", fn)

summary = {}
for tag, ls, lt, floor in RUNS:
    if ONLY_RUN and tag != ONLY_RUN:
        continue
    out = run_all(tag, ls, lt, floor)
    sc = score(out["res"]); summary[tag] = (sc, out["fit"])
    figures(tag, out["res"])

for tag, (sc, fit) in summary.items():
    print("\n" + "=" * 96)
    print(f"[{tag}]  Injection-zone recovery (x {ZONE['x0']}-{ZONE['x1']} m, top {-ZONE['zmin']} m).  "
          "Truth: strong DECREASE toward ~10 Ohm.m after t=4 h")
    print("=" * 96)
    print(f"{'method':16s}" + "".join(f"{'t='+str(t)+'h':>7s}" for t in TIMES)
          + f"{'zone chg':>10s}{'zone min':>10s}{'bg chg':>8s}{'fit0':>7s}{'fit':>7s}{'rough':>7s}")
    for m in METHODS:
        r = sc[m]
        print(f"{m:16s}" + "".join(f"{v:7.0f}" for v in r["zone"])
              + f"{r['zchg']:+9.0f}%{r['zmin']:10.0f}{r['bchg']:+7.0f}%"
              + f"{fit[m][0]:7.1f}{np.nanmean(fit[m][1:]):7.1f}{r['rough']:7.2f}")
    print("zone chg = median zone resistivity, mean(t=4-24 h) vs t=0 (negative = plume recovered);")
    print("zone min = most conductive cell in the zone after injection; bg chg = same away from it;")
    print("fit0 = data log-RMS (%) at t=0; fit = mean for t>=4 h (flat 190 Ohm.m ground scores ~68%);")
    print("rough = temporal roughness.")
print("\nDone.")
