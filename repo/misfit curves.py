# -*- coding: utf-8 -*-
"""R3-6: per-epoch data misfit for Models 1 and 2, on statistics that mean
what they look like they mean.

The mean relative RMS is unusable here.  Dipole-dipole readings over a strong
conductor drop below one ohm-metre, and dividing by them makes the mean blow
up - forward-modelling the true model scores between 43 and 337 per cent on
it.  A perfect model cannot be wrong by 300 per cent, so that number is
measuring its own denominators.

Plotted instead: chi-squared against the assumed 5 per cent floor, which is
what the inversion actually minimises and should sit near 1, and the median
absolute relative error, which ignores the outliers and still reads in per
cent.  The true model is forward-modelled on the inversion mesh and drawn on
both panels.  That curve is the control: if it also scores in the hundreds the
statistic is broken, and if it scores a few per cent while a method scores
hundreds then that method really is leaving the data unfitted.

Nothing is inverted here.  The models come from the padded-domain caches, so
the cost is one forward solve per model per epoch.

Writes model12_data_misfit_v2.png and prints the same numbers.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.interpolate import griddata

import os, pickle, time
from scipy.interpolate import griddata

OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
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
fwd_new = forward_mesh(120.0, 80.0)                    # padded (~10x spread)
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





# --- the figure: chi-squared and median relative error, with the true
#     model on both panels as the reference no inversion can beat -------
METHOD_COLORS = {"Independent":   "#1f77b4",
                 "Difference":    "#ff7f0e",
                 "4D L2-Coupled": "#d62728",
                 "HWSI":          "#2ca02c",
                 "TRUE MODEL":    "#444444"}

def forward_response(model, data):
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(inv_mesh)
    return np.maximum(np.array(fop.response(np.asarray(model, float))), 1e-6)

para_cc = np.array([[c.center().x(), c.center().y()] for c in para_mesh.cells()])
ROWS = list(METHODS) + ["TRUE MODEL"]
chi2, medrel = {}, {}

for mname in MODELS_TO_RUN:
    cache = os.path.join(OUT, f"padded_{mname.replace(' ', '').lower()}.pkl")
    if not os.path.exists(cache):
        print(f"[{mname}] no cache - skipping"); continue
    cached = pickle.load(open(cache, "rb"))
    point_fn, grid_fn = DEF[mname]
    print(f"\n[{mname}] forward responses ...")

    data_list, true_para = [], []
    for t in times:
        rv = np.array([point_fn(c.center().x(), c.center().y(), t)
                       for c in fwd_new.cells()])
        d = ert.simulate(mesh=fwd_new, scheme=scheme, res=rv, noiseLevel=NOISE,
                         seed=123, verbose=False)
        d["rhoa"] = np.maximum(np.abs(d["rhoa"]), 0.1)
        d.set("err", np.full(d.size(), FLOOR))
        data_list.append(d)
        true_para.append(np.array([point_fn(x, z, t) for x, z in para_cc]))

    chi2[mname], medrel[mname] = {}, {}
    for m in ROWS:
        c_curve, r_curve = [], []
        for ti in range(len(times)):
            mod  = true_para[ti] if m == "TRUE MODEL" else cached["models"][m][ti]
            obs  = np.array(data_list[ti]["rhoa"], float)
            pred = forward_response(mod, data_list[ti])
            c_curve.append(float(np.mean(((np.log(obs) - np.log(pred)) / FLOOR) ** 2)))
            r_curve.append(float(np.median(np.abs((obs - pred) / obs)) * 100))
        chi2[mname][m], medrel[mname][m] = c_curve, r_curve
        print(f"  {m:<16} chi2 " + " ".join(f"{v:8.1f}" for v in c_curve))
        print(f"  {'':<16} med% " + " ".join(f"{v:8.2f}" for v in r_curve))

if chi2:
    names = list(chi2)
    fig, axes = plt.subplots(2, len(names), figsize=(7.0 * len(names), 9.0),
                             squeeze=False)
    panels = [(chi2,   r"$\chi^2$  (5% error floor)",      True),
              (medrel, "median |relative error|  (%)",     False)]
    for k, mname in enumerate(names):
        for row, (data, lab, logy) in enumerate(panels):
            ax = axes[row][k]
            for m in ROWS:
                st = dict(linewidth=2.2, marker="o", markersize=5)
                if m == "TRUE MODEL":
                    st.update(linewidth=2.0, linestyle="--", marker="s")
                ax.plot(times, data[mname][m], label=m, color=METHOD_COLORS[m], **st)
            if logy:
                ax.set_yscale("log")
                ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.4)
                ax.text(times[-1], 1.0, " target", va="bottom", ha="right",
                        fontsize=9, color="0.35")
            ax.set_xlabel("time (h)", fontweight="bold")
            if k == 0:
                ax.set_ylabel(lab, fontweight="bold")
            if row == 0:
                ax.set_title(mname, fontweight="bold")
            ax.set_xticks(list(times))
            ax.grid(True, alpha=0.3, linestyle="--")
            if row == 0 and k == 0:
                ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT, "model12_data_misfit_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nsaved -> {out}")
print("\nDone.")
