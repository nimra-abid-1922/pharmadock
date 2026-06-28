import os

_here = os.path.dirname(os.path.abspath(__file__))

_on_server = os.path.exists("/root/data/targets.json")

INPUT_PATH = "/root/data/targets.json" if _on_server else os.path.join(_here, "targets.json")
OUTPUT_PATH = "/root/results/docked_poses.sdf" if _on_server else os.path.join(_here, "results", "docked_poses.sdf")

EXCL_RADIUS = 1.2 #default from the task spec when the JSON entry has no explicit radius
CLASH_TOL = 0.1 #from the task spec; effective hard boundary is 1.2 - 0.1 = 1.1 Å
SCORE_SIGMA = 1.25 # directly from the task spec scoring formula

BASE_CONFS = 96
CONFS_PER_ROT = 80
MAX_CONFS = 800 # cap to avoid memory issues on very flexible molecules
PRUNE_RMS = 0.25
PRUNE_RMS_LOOSE = 0.75  #rigid molecules barely pass the tight threshold so we retry with this

DIVERSE_RMSD = 2.0
DIVERSE_PRESELECT = 500

MAX_PAIRS_PER_SITE = 16 #keeps the triplet pair matrix from exploding on large flexible molecules
TRIPLET_TOL = 1.6 # Å,roughly one bond length
MAX_ENUM = 8000 # clique search can blow up on dense graphs
SEEDS_PER_CONF = 55
ROTS_PER_SITE = 30
RAND_ROTS = 150

TOP_K = 50
FLEX_K = 20
FLEX_HOPS = 12
FLEX_ROT_SIGMA = 0.4 # around 23 degrees per hop
FLEX_TRANS_SIGMA = 0.6
FLEX_TORS_SIGMA = 1.2  #around 69 degrees, enough to cross torsion barriers
FLEX_ACCEPT_PROB = 0.3
MAX_TORSIONS = 12  #Powell struggles to converge beyond this many free variables

COORDMAP_ANCHORS = 3
COORDMAP_ATOMS = 4
PENALTY = 500.0 # must dominate the pharmacophore score to reliably enforce exclusion

SITE_FAM_TO_RDKIT = {
    "Donor": {"Donor"},
    "Acceptor": {"Acceptor"},
    "Hydrophobe": {"Hydrophobe", "LumpedHydrophobe"},
    "Aromatic": {"Aromatic"},
}
