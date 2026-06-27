import json, os, math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

_here       = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = "/root/data/targets.json" if os.path.exists("/root/data/targets.json") else os.path.join(_here, "targets.json")
OUTPUT_PATH = "/root/results/docked_poses.sdf" if os.path.exists("/root/data/targets.json") else os.path.join(_here, "results", "docked_poses.sdf")

EXCL_RADIUS        = 1.2
CLASH_TOL          = 0.1
SCORE_SIGMA        = 1.25
NUM_CONFS          = 200
PRUNE_RMS          = 0.3
MAX_PAIRS_PER_SITE = 14
TRIPLET_TOL        = 1.5
MAX_ENUM           = 6000
SEEDS_PER_CONF     = 40
ROTS_PER_SITE      = 24
RAND_ROTS          = 120
TOP_K              = 45
PENALTY            = 50.0

SITE_FAM_TO_RDKIT = {
    "Donor":      {"Donor"},
    "Acceptor":   {"Acceptor"},
    "Hydrophobe": {"Hydrophobe", "LumpedHydrophobe"},
    "Aromatic":   {"Aromatic"},
}

_feat_factory = None

def feat_factory():
    global _feat_factory
    if _feat_factory is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _feat_factory

def pharmacophore_atoms(mol):
    feats = feat_factory().GetFeaturesForMol(mol)
    groups = {fam: set() for fam in SITE_FAM_TO_RDKIT}
    for feat in feats:
        rdkit_fam = feat.GetFamily()
        for site_fam, rdkit_fams in SITE_FAM_TO_RDKIT.items():
            if rdkit_fam in rdkit_fams:
                groups[site_fam].update(feat.GetAtomIds())
    return {fam: sorted(idxs) for fam, idxs in groups.items()}

def score_pose(atom_pos, sites, pharm_groups):
    total = 0.0
    for site in sites:
        idxs = pharm_groups.get(site["family"], [])
        if not idxs:
            continue
        site_pos = np.array([site["x"], site["y"], site["z"]])
        d_min = np.linalg.norm(atom_pos[idxs] - site_pos, axis=1).min()
        total += site["weight"] * math.exp(-(d_min / SCORE_SIGMA) ** 2)
    return total

def clash_amount(atom_pos, excl_vols):
    overlap = 0.0
    for ev in excl_vols:
        center = np.array([ev["x"], ev["y"], ev["z"]])
        threshold = ev.get("radius", EXCL_RADIUS) - CLASH_TOL
        d_min = np.linalg.norm(atom_pos - center, axis=1).min()
        if d_min < threshold:
            overlap += threshold - d_min
    return overlap

def steric_clash(atom_pos, excl_vols):
    return clash_amount(atom_pos, excl_vols) > 0.0

def kabsch(source, target):
    sc, tc = source.mean(0), target.mean(0)
    H = (source - sc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, det]) @ U.T
    t = tc - sc @ R.T
    return R, t

def rand_rot(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)     ],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)     ],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y) ],
    ])

def neg_objective(params, centered, sites, pharm_groups, excl_vols):
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    pos = centered @ R.T + params[3:]
    return -(score_pose(pos, sites, pharm_groups) - PENALTY * clash_amount(pos, excl_vols))

def refine(pos, sites, pharm_groups, excl_vols):
    center   = pos.mean(0)
    centered = pos - center
    x0       = np.concatenate([np.zeros(3), center])
    res = minimize(
        neg_objective, x0,
        args=(centered, sites, pharm_groups, excl_vols),
        method="Powell",
        options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-4},
    )
    R = Rotation.from_rotvec(res.x[:3]).as_matrix()
    return centered @ R.T + res.x[3:]

def matched_pairs(sites, pharm_groups, rng):
    s_idx, a_idx, s_pos = [], [], []
    for si, site in enumerate(sites):
        idxs = pharm_groups.get(site["family"], [])
        if not idxs:
            continue
        if len(idxs) > MAX_PAIRS_PER_SITE:
            idxs = list(rng.choice(idxs, MAX_PAIRS_PER_SITE, replace=False))
        sp = [site["x"], site["y"], site["z"]]
        for a in idxs:
            s_idx.append(si)
            a_idx.append(a)
            s_pos.append(sp)
    return np.array(s_idx), np.array(a_idx), np.array(s_pos, float)

def triplet_seeds(raw, s_idx, a_idx, s_pos):
    n = len(s_idx)
    if n < 3:
        return []
    atom_pts = raw[a_idx]
    d_site = cdist(s_pos, s_pos)
    d_atom = cdist(atom_pts, atom_pts)
    adj = np.abs(d_site - d_atom) < TRIPLET_TOL
    adj &= s_idx[:, None] != s_idx[None, :]
    adj &= a_idx[:, None] != a_idx[None, :]
    np.fill_diagonal(adj, False)

    neighbors = [np.where(row)[0] for row in adj]
    tris = []
    for i in range(n):
        ni = neighbors[i]
        for j in ni[ni > i]:
            common = np.intersect1d(ni, neighbors[j])
            for k in common[common > j]:
                err = (abs(d_site[i, j] - d_atom[i, j]) +
                       abs(d_site[i, k] - d_atom[i, k]) +
                       abs(d_site[j, k] - d_atom[j, k]))
                tris.append((err, i, j, k))
        if len(tris) > MAX_ENUM:
            break
    if not tris:
        return []

    tris.sort(key=lambda t: t[0])
    seeds = []
    for _, i, j, k in tris[:SEEDS_PER_CONF]:
        R, t = kabsch(atom_pts[[i, j, k]], s_pos[[i, j, k]])
        seeds.append(raw @ R.T + t)
    return seeds

def conformer_candidates(raw, sites, pharm_groups, pharm_center, excl_vols, rng):
    s_idx, a_idx, s_pos = matched_pairs(sites, pharm_groups, rng)
    candidates = []

    def consider(pos):
        if not steric_clash(pos, excl_vols):
            candidates.append((score_pose(pos, sites, pharm_groups), pos.copy()))

    for pos in triplet_seeds(raw, s_idx, a_idx, s_pos):
        consider(pos)

    for site in sites:
        idxs = pharm_groups.get(site["family"], [])
        if not idxs:
            continue
        site_pos  = np.array([site["x"], site["y"], site["z"]])
        rots_each = max(1, ROTS_PER_SITE // len(idxs))
        for atom_idx in idxs:
            anchored = raw - raw[atom_idx]
            for _ in range(rots_each):
                consider(anchored @ rand_rot(rng).T + site_pos)

    centered = raw - raw.mean(0)
    for _ in range(RAND_ROTS):
        consider(centered @ rand_rot(rng).T + pharm_center)

    return candidates

def embed_conformers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed   = 42
    params.pruneRmsThresh = PRUNE_RMS
    AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return Chem.RemoveHs(mol)

def dock(smiles, sites, excl_vols):
    mol = embed_conformers(smiles)
    if mol.GetNumConformers() == 0:
        return mol, None

    pharm_groups = pharmacophore_atoms(mol)
    pharm_center = np.mean([[s["x"], s["y"], s["z"]] for s in sites], axis=0)

    rng  = np.random.default_rng(42)
    pool = []
    for ci in range(mol.GetNumConformers()):
        raw = np.array(mol.GetConformer(ci).GetPositions())
        pool.extend(
            conformer_candidates(raw, sites, pharm_groups, pharm_center, excl_vols, rng)
        )

    if not pool:
        return mol, None

    pool.sort(key=lambda c: c[0], reverse=True)
    best_score, best_pos = pool[0]
    for _, pos in pool[:TOP_K]:
        refined = refine(pos, sites, pharm_groups, excl_vols)
        if steric_clash(refined, excl_vols):
            continue
        s = score_pose(refined, sites, pharm_groups)
        if s > best_score:
            best_score, best_pos = s, refined

    return mol, best_pos

def build_output_mol(mol, positions):
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos.tolist())
    out = Chem.RWMol(mol)
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out.GetMol()

def main():
    with open(INPUT_PATH) as fh:
        targets = json.load(fh)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = Chem.SDWriter(OUTPUT_PATH)

    for name, target in targets.items():
        mol, positions = dock(
            target["smiles"],
            target["interaction_sites"],
            target.get("excluded_volumes", []),
        )
        if positions is None:
            continue
        out_mol = build_output_mol(mol, positions)
        out_mol.SetProp("_Name", name)
        writer.write(out_mol)

    writer.close()

if __name__ == "__main__":
    main()
