History-Weighted Spatiotemporal Inversion (HWSI): revision code
Code accompanying the revised manuscript

Singh, G., Banerjee Chattopadhyay, P., Gupta, G., Kumar, P. History-Weighted Spatiotemporal Inversion (HWSI): A Cell-Adaptive Temporal Regularisation Framework for Time-Lapse Electrical Resistivity Tomography. Computers & Geosciences (submitted, CAGEO-D-26-00801).

HWSI gives every model cell its own temporal coupling coefficient, derived from how much that cell changed over the preceding transition, and refreshes the coefficients at every Gauss–Seidel sweep. Each epoch is coupled only to its predecessor, so memory stays that of a single-epoch inversion.

This repository contains the experiments added in the revision: the padded-domain check, the data-misfit curves, the Model 2 sensitivity sweeps, the electrode-array study and the saline-tracer field validation. The original implementation and the Model 1 / Model 2 benchmarks (Tables 3–8, Figs 2–13) are in the first-version repository: https://github.com/GarvitG19/History-Weighted_Spatiotemporal_Inversion_HWSI

Installation
conda env create -f environment.yml
conda activate hwsi
or, with an existing Python 3.9+ environment, pip install -r requirements.txt.

pyGIMLi 1.5.5 is pinned on purpose. Later releases change the pygimli.physics.ert interface (TimelapseERT, ERTManager.invert), and the results are not guaranteed to reproduce with them.

Which script produces which result
Every script is self-contained: the HWSI engine and the three benchmark methods are bundled in each file, so a script can be run on its own from any folder. Outputs (.pkl caches and .png figures) are written next to the script.

Script	Manuscript result	Notes
padded_full_domain.py	Table 10; Sections 4.1 and 5.2	Forward domain padded to x = ±120 m, z = −80 m; masked and full-domain errors; homogeneous half-space boundary check. Writes padded_model1.pkl, padded_model2.pkl.
misfit curves.py	Fig. 17	Per-epoch χ² and median relative error for Models 1 and 2. Run padded_full_domain.py first: it reads the two padded_*.pkl caches.
senstivity and noise analysis model 2.py	Table 11 and Fig. 14 (DO_LAMBDA_S = True); Table 12 and Fig. 15 (DO_NOISE = True)	Model 2, λs = 0.1–1000 and matched noise/error floor 1–7 %. To reproduce the published λs column for HWSI exactly, set HWSI_RELAX = 1.0 and HWSI_BEST = False. FAST = True gives the trends in about 20–35 min on a coarse mesh, but not the published numbers.
array_sensitivity.py	Table 13 and Fig. 16	Models 1 and 2 with dipole-dipole, Wenner and Schlumberger arrays; all other settings identical.
schlumberger_stability_check.py	Fig. 16 caption (Model 1, Schlumberger)	Repeats the one case in which HWSI did not settle, with stronger under-relaxation and more sweeps.
real_data_hwsi.py	Section 8, Table 14, Figs 18–19	Saline-tracer field experiment, run at λs = λt = 20 (5 % floor) and λs = 80, λt = 30 (8 % floor); HWSI from independent and from 4D seeds. Requires the field data (see below).
The sensitivity sweeps use the fine paper meshes and take several hours each on a standard workstation. All synthetic data use a fixed noise seed (123), so repeated runs on the same software versions give the same numbers.

Note on equation numbers: comments in the scripts refer to the nRMSE as "Eq. 11", its number in the first submission. In the revised manuscript it is Eq. (17).

Field data
The field data are seven Res2DInv general-array files (2.Dat, 3.Dat, 4.h5new.Dat, 5.Dat … 8.Dat; epochs 0–24 h at 4 h intervals; 24 electrodes at 0.5 m; dipole-dipole). They are available from the corresponding author on reasonable request. Place them in a folder called field_data/ next to real_data_hwsi.py, or point the environment variable HWSI_FIELD_DATA to their folder.

Citation
Please cite the article above if you use this code. GitHub's "Cite this repository" button uses CITATION.cff.

Licence and contact
MIT licence (see LICENSE). Corresponding author: Pallavi Banerjee Chattopadhyay, Department of Earth Sciences, IIT Roorkee (cpallavi@es.iitr.ac.in).

