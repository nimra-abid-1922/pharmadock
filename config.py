import os

_here = os.path.dirname(os.path.abspath(__file__))

_on_server = os.path.exists("/root/data/targets.json")

INPUT_PATH = "/root/data/targets.json" if _on_server else os.path.join(_here, "targets.json")
OUTPUT_PATH = "/root/results/docked_poses.sdf" if _on_server else os.path.join(_here, "results", "docked_poses.sdf")

EXCL_RADIUS = 1.2  #input exclusion spheres overlap heavily
CLASH_TOL = 0.1
SCORE_SIGMA = 1.25

BASE_CONFS = 96
CONFS_PER_ROT = 80
MAX_CONFS = 800
PRUNE_RMS = 0.25
PRUNE_RMS_LOOSE = 0.75

DIVERSE_RMSD = 2.0
DIVERSE_PRESELECT = 500

MAX_PAIRS_PER_SITE = 16
TRIPLET_TOL = 1.6  # Å
MAX_ENUM = 8000
SEEDS_PER_CONF = 55
ROTS_PER_SITE = 30
RAND_ROTS = 150

TOP_K = 50
FLEX_K = 20
FLEX_HOPS = 12
FLEX_ROT_SIGMA = 0.4
FLEX_TRANS_SIGMA = 0.6
FLEX_TORS_SIGMA = 1.2
FLEX_ACCEPT_PROB = 0.3
MAX_TORSIONS = 12

COORDMAP_ANCHORS = 3
COORDMAP_ATOMS = 4
PENALTY = 500.0  # must dominate pharmacophore score to keep poses out of excluded volumes

SITE_FAM_TO_RDKIT = {
    "Donor": {"Donor"},
    "Acceptor": {"Acceptor"},
    "Hydrophobe": {"Hydrophobe", "LumpedHydrophobe"},
    "Aromatic": {"Aromatic"},
}
