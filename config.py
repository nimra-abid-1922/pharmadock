import os

_here = os.path.dirname(os.path.abspath(__file__))

_on_server = os.path.exists("/root/data/targets.json")

INPUT_PATH = "/root/data/targets.json" if _on_server else os.path.join(_here, "targets.json")
OUTPUT_PATH = "/root/results/docked_poses.sdf" if _on_server else os.path.join(_here, "results", "docked_poses.sdf")

EXCL_RADIUS = 1.2  #default from the task spec when the JSON entry has no explicit radius
CLASH_TOL = 0.1  # tolerance from the task spec; effective hard boundary is 1.2 - 0.1 = 1.1 Å
SCORE_SIGMA = 1.25  # directly from the task spec scoring formula

BASE_CONFS = 96  # baseline conformer count before scaling up for flexible molecules
CONFS_PER_ROT = 80  #each rotatable bond meaningfully expands the conformational space
MAX_CONFS = 800  # cap to avoid memory issues on very flexible molecules
PRUNE_RMS = 0.25  # filters near-duplicate conformers without throwing away genuine diversity
PRUNE_RMS_LOOSE = 0.75  # rigid molecules barely pass the tight threshold so we retry with this

DIVERSE_RMSD = 2.0  #poses closer than this are considered the same region and we skip the second
DIVERSE_PRESELECT = 500  # only the top N candidates go into the diversity filter; the rest score too low to matter

MAX_PAIRS_PER_SITE = 16  # keeps the triplet pair matrix from exploding on large flexible molecules
TRIPLET_TOL = 1.6 #Å, roughly one bond length; tighter misses real matches, looser brings in too much noise
MAX_ENUM = 8000  #the clique search can explode on dense graphs so we cut it off here
SEEDS_PER_CONF = 55  # best-matching triplets used to generate starting poses per conformer
ROTS_PER_SITE = 30  # random rotations tried per atom anchored on a site
RAND_ROTS = 150  # fallback random spins to cover geometry the triplets might have missed

TOP_K = 50  # how many coarse candidates get a full rigid Powell refinement pass
FLEX_K = 20  # the top N of those also get flexible bond refinement on top
FLEX_HOPS = 12  # basin hopping iterations per flexible refinement
FLEX_ROT_SIGMA = 0.4  # about 23 degrees of rotation noise per hop
FLEX_TRANS_SIGMA = 0.6  # a of translation noise per hop
FLEX_TORS_SIGMA = 1.2  # about 69 degrees of torsion noise per hop, enough to cross barriers
FLEX_ACCEPT_PROB = 0.3  # roughly 1 in 3 worse hops are accepted to escape local minima
MAX_TORSIONS = 12  # more than this and Powell can't converge within a reasonable time budget

COORDMAP_ANCHORS = 3  # number of top-weight sites to pin when generating coordMap-guided conformers
COORDMAP_ATOMS = 4  # atom combinations tried per anchor site
PENALTY = 500.0  #must dominate the pharmacophore score to reliably keep poses out of excluded volumes

SITE_FAM_TO_RDKIT = {
    "Donor": {"Donor"},
    "Acceptor": {"Acceptor"},
    "Hydrophobe": {"Hydrophobe", "LumpedHydrophobe"},
    "Aromatic": {"Aromatic"},
}
