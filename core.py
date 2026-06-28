import json, os, math, io
from itertools import product as iprod
from contextlib import redirect_stdout
import multiprocessing as mp
import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors
from rdkit.Geometry.rdGeometry import Point3D
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from config import (
    INPUT_PATH, OUTPUT_PATH,
    EXCL_RADIUS, CLASH_TOL, SCORE_SIGMA,
    BASE_CONFS, CONFS_PER_ROT, MAX_CONFS, PRUNE_RMS, PRUNE_RMS_LOOSE,
    MAX_PAIRS_PER_SITE, TRIPLET_TOL, MAX_ENUM, SEEDS_PER_CONF,
    ROTS_PER_SITE, RAND_ROTS, TOP_K, FLEX_K, FLEX_HOPS,
    FLEX_ROT_SIGMA, FLEX_TRANS_SIGMA, FLEX_TORS_SIGMA, FLEX_ACCEPT_PROB,
    MAX_TORSIONS, COORDMAP_ANCHORS, COORDMAP_ATOMS, PENALTY,
    DIVERSE_RMSD, DIVERSE_PRESELECT,
    SITE_FAM_TO_RDKIT,
)

_feat_factory = None  #re-parsing BaseFeatures.fdef on every molecule call adds measurable overhead at scale


def feat_factory():
    global _feat_factory
    if _feat_factory is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _feat_factory


def pharmacophore_atoms(mol):
    feats = feat_factory().GetFeaturesForMol(mol)
    by_family = {fam: set() for fam in SITE_FAM_TO_RDKIT}
    for feat in feats:
        rdkit_fam = feat.GetFamily()
        for site_fam, rdkit_fams in SITE_FAM_TO_RDKIT.items():
            if rdkit_fam in rdkit_fams:
                by_family[site_fam].update(feat.GetAtomIds())
    return {fam: sorted(idxs) for fam, idxs in by_family.items()}


def score_pose(atom_pos, sites, pharm_groups):
    score = 0.0
    for site in sites:
        idxs = pharm_groups.get(site["family"], [])
        if not idxs:
            continue
        sp = np.array([site["x"], site["y"], site["z"]])
        d_min = np.linalg.norm(atom_pos[idxs] - sp, axis=1).min()
        score += site["weight"] * math.exp(-(d_min / SCORE_SIGMA) ** 2)
    return score


def clash_amount(atom_pos, excl_vols):
    if not excl_vols:
        return 0.0
    total = 0.0
    for ev in excl_vols:
        center = np.array([ev["x"], ev["y"], ev["z"]])
        threshold = ev.get("radius", EXCL_RADIUS) - CLASH_TOL
        d_min = np.linalg.norm(atom_pos - center, axis=1).min()
        if d_min < threshold:
            total += threshold - d_min
    return total


def steric_clash(atom_pos, excl_vols):
    return clash_amount(atom_pos, excl_vols) > 0.0


def kabsch(source, target):
    sc, tc = source.mean(0), target.mean(0)
    H = (source - sc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    #det flips the sign to avoid a reflection when the point sets are mirror images
    R = Vt.T @ np.diag([1.0, 1.0, det]) @ U.T
    t = tc - sc @ R.T
    return R, t


def pose_loss(params, centered, sites, pharm_groups, excl_vols):
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    pos = centered @ R.T + params[3:]
    return -(score_pose(pos, sites, pharm_groups) - PENALTY * clash_amount(pos, excl_vols))  # Powell minimizes so we negate


def refine(pos, sites, pharm_groups, excl_vols):
    center = pos.mean(0)
    centered = pos - center
    x0 = np.zeros(6)
    x0[3:] = center
    res = minimize(
        pose_loss, x0,
        args=(centered, sites, pharm_groups, excl_vols),
        method="Powell",
        options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-4},
    )
    R = Rotation.from_rotvec(res.x[:3]).as_matrix()
    return centered @ R.T + res.x[3:]


def side_atoms(mol, a, b):
    seen = {a}
    stack = [b]
    comp = []
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        comp.append(x)
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            j = nb.GetIdx()
            if j != a and j not in seen:
                stack.append(j)
    return comp


def rotatable_torsions(mol):
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    torsions = []
    seen_bonds = set() #the SMARTS pattern returns each bond twice, once per direction
    for a, b in mol.GetSubstructMatches(patt):
        bond_key = (min(a, b), max(a, b))
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)
        moving = side_atoms(mol, a, b)
        if len(moving) > mol.GetNumAtoms() - len(moving):
            a, b = b, a
            moving = side_atoms(mol, a, b) #always rotate the smaller fragment to keep the inner loop cheap
        torsions.append((a, b, np.array(moving)))
    return torsions[:MAX_TORSIONS]


def apply_torsions(base, params, torsions):
    pos = base.copy()
    angles = params[6:]
    for (a, b, moving), angle in zip(torsions, angles):
        axis = pos[b] - pos[a]
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        R = Rotation.from_rotvec(axis / n * angle).as_matrix()
        pos[moving] = (pos[moving] - pos[b]) @ R.T + pos[b]
    center = pos.mean(0)
    rot = Rotation.from_rotvec(params[:3]).as_matrix()
    return (pos - center) @ rot.T + center + params[3:6]


def flexible_refine(base, torsions, sites, pharm_groups, excl_vols, rng):
    n = len(torsions)
    dim = 6 + n

    def objective(x):
        pos = apply_torsions(base, x, torsions)
        return -(score_pose(pos, sites, pharm_groups) - PENALTY * clash_amount(pos, excl_vols))

    def local(start):
        res = minimize(objective, start, method="Powell",
                       options={"maxiter": 2500, "xtol": 1e-3, "ftol": 1e-3})
        return res.x, res.fun

    best_x, best_f = local(np.zeros(dim))
    cur_x, cur_f = best_x.copy(), best_f

    for _ in range(FLEX_HOPS):
        step = np.zeros(dim)
        step[:3] = rng.normal(0, FLEX_ROT_SIGMA, 3)
        step[3:6] = rng.normal(0, FLEX_TRANS_SIGMA, 3)
        if n:
            step[6:] = rng.normal(0, FLEX_TORS_SIGMA, n)
        cand_x, cand_f = local(cur_x + step)
        if cand_f < best_f:
            best_x, best_f = cand_x, cand_f
        if cand_f < cur_f or rng.random() < FLEX_ACCEPT_PROB:
            cur_x, cur_f = cand_x, cand_f #occasional uphill moves let the search escape local basins

    return apply_torsions(base, best_x, torsions)


def matched_pairs(sites, pharm_groups, rng):
    s_idx, a_idx, s_pos = [], [], []
    for si, site in enumerate(sites):
        fam = site["family"]
        idxs = pharm_groups.get(fam, [])
        if not idxs:
            continue
        if len(idxs) > MAX_PAIRS_PER_SITE:
            idxs = list(rng.choice(idxs, MAX_PAIRS_PER_SITE, replace=False))
        sp = [site["x"], site["y"], site["z"]]
        for a in idxs:
            s_idx.append(si)
            a_idx.append(a)
            s_pos.append(sp)
    if not s_idx:
        return np.array([]), np.array([]), np.zeros((0, 3))
    return np.array(s_idx), np.array(a_idx), np.array(s_pos, float)


def triplet_seeds(raw, s_idx, a_idx, s_pos):
    n = len(s_idx)
    if n < 3:
        return []
    atom_pts = raw[a_idx]
    d_site = cdist(s_pos, s_pos)
    d_atom = cdist(atom_pts, atom_pts)

    compatible = np.abs(d_site - d_atom) < TRIPLET_TOL
    compatible &= s_idx[:, None] != s_idx[None, :]
    compatible &= a_idx[:, None] != a_idx[None, :]
    np.fill_diagonal(compatible, False)

    neighbors = [np.where(row)[0] for row in compatible]
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
            break #clique enumeration can explode on densely connected pair graphs
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

    for pos in triplet_seeds(raw, s_idx, a_idx, s_pos):
        if not steric_clash(pos, excl_vols):
            candidates.append((score_pose(pos, sites, pharm_groups), pos.copy()))

    for site in sites:
        idxs = pharm_groups.get(site["family"], [])
        if not idxs:
            continue
        site_pos = np.array([site["x"], site["y"], site["z"]])
        rots_each = max(1, ROTS_PER_SITE // len(idxs))
        for atom_idx in idxs:
            anchored = raw - raw[atom_idx]
            for _ in range(rots_each):
                pos = anchored @ Rotation.random(random_state=rng).as_matrix().T + site_pos
                if not steric_clash(pos, excl_vols):
                    candidates.append((score_pose(pos, sites, pharm_groups), pos.copy()))

    centered = raw - raw.mean(0)
    for _ in range(RAND_ROTS):
        pos = centered @ Rotation.random(random_state=rng).as_matrix().T + pharm_center
        if not steric_clash(pos, excl_vols):
            candidates.append((score_pose(pos, sites, pharm_groups), pos.copy()))

    return candidates


def coordmap_candidates(mol_noh, sites, pharm_groups, excl_vols):
    sorted_si = sorted(range(len(sites)), key=lambda i: -sites[i]["weight"])
    anchors = sorted_si[:COORDMAP_ANCHORS]
    atom_choices = []
    for si in anchors:
        atoms = pharm_groups.get(sites[si]["family"], [])
        atom_choices.append(atoms[:COORDMAP_ATOMS] if atoms else [])

    if any(len(ch) == 0 for ch in atom_choices):
        return []

    candidates = []
    for combo in iprod(*atom_choices):
        coord_map = {
            atom_idx: Point3D(sites[si]["x"], sites[si]["y"], sites[si]["z"])
            for si, atom_idx in zip(anchors, combo)
        }
        mol = Chem.AddHs(mol_noh)
        cid = -1
        for seed in range(8):
            cid = AllChem.EmbedMolecule(mol, coordMap=coord_map, randomSeed=seed)
            if cid != -1:
                break
        if cid == -1:
            continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=600)
        pos = np.array(mol.GetConformer().GetPositions()[:mol_noh.GetNumAtoms()])
        if not steric_clash(pos, excl_vols):
            candidates.append((score_pose(pos, sites, pharm_groups), pos.copy()))
    return candidates


def diverse_top_k(pool, k):
    #top-K by score alone clusters around one region; RMSD filter spreads the refinement budget
    if len(pool) <= k:
        return pool[:]
    candidates = pool[:DIVERSE_PRESELECT]
    selected = []
    seen_pos = []
    taken = set()
    for i, (score, pos) in enumerate(candidates):
        if not any(np.sqrt(((pos - sp) ** 2).mean()) < DIVERSE_RMSD for sp in seen_pos):
            selected.append((score, pos))
            seen_pos.append(pos)
            taken.add(i)
        if len(selected) >= k:
            return selected
    for i, item in enumerate(candidates):
        if i not in taken:
            selected.append(item)
        if len(selected) >= k:
            break
    return selected


def embed_conformers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse SMILES: {smiles}"
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_confs = min(MAX_CONFS, BASE_CONFS + CONFS_PER_ROT * rot)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = PRUNE_RMS
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if mol.GetNumConformers() < 10:
        #tight RMS pruning wipes out almost everything for rigid or small molecules so we retry with a looser threshold
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        params.pruneRmsThresh = PRUNE_RMS_LOOSE
        AllChem.EmbedMultipleConfs(mol, numConfs=max(n_confs, 100), params=params)

    if mol.GetNumConformers() < 2:
        #last resort for molecules that fail distance geometry entirely
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 0
        params.pruneRmsThresh = 0.0
        AllChem.EmbedMultipleConfs(mol, numConfs=max(n_confs, 50), params=params)

    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return Chem.RemoveHs(mol)


def dock(smiles, sites, excl_vols, label="", verbose=True):
    if not sites:
        return Chem.MolFromSmiles(smiles), None, 0.0, 0.0

    tag = f"  [{label}]" if label else " "

    if verbose:
        print(f"{tag} generating conformers...")
    mol = embed_conformers(smiles)
    if mol.GetNumConformers() == 0:
        return mol, None, 0.0, sum(s["weight"] for s in sites)
    if verbose:
        print(f"{tag} {mol.GetNumConformers()} conformers ready.")

    pharm_groups = pharmacophore_atoms(mol)
    site_coords = [[s["x"], s["y"], s["z"]] for s in sites]
    pharm_center = np.array(site_coords).mean(axis=0)
    torsions = rotatable_torsions(mol)

    if verbose:
        print(f"{tag} building candidate pool...")
    mx = sum(s["weight"] for s in sites)
    rng = np.random.default_rng(42)
    pool = list(coordmap_candidates(mol, sites, pharm_groups, excl_vols))
    for ci in range(mol.GetNumConformers()):
        raw = np.array(mol.GetConformer(ci).GetPositions())
        pool.extend(conformer_candidates(raw, sites, pharm_groups, pharm_center, excl_vols, rng))

    if not pool:
        return mol, None, 0.0, mx

    pool.sort(key=lambda c: c[0], reverse=True)
    best_score, best_pos = pool[0]
    if verbose:
        print(f"{tag} {len(pool)} candidates, top coarse score {best_score:.3f}.")
        print(f"{tag} refining top {TOP_K} rigid + {FLEX_K} flexible...")

    for rank, (_, coarse) in enumerate(diverse_top_k(pool, TOP_K)):
        refined = refine(coarse, sites, pharm_groups, excl_vols)
        rigid_ok = not steric_clash(refined, excl_vols)
        if rigid_ok:
            s = score_pose(refined, sites, pharm_groups)
            if s > best_score:
                best_score, best_pos = s, refined

        if rank < FLEX_K:
            flex_base = refined if rigid_ok else coarse
            flexed = flexible_refine(flex_base, torsions, sites, pharm_groups, excl_vols, rng)
            if not steric_clash(flexed, excl_vols):
                s = score_pose(flexed, sites, pharm_groups)
                if s > best_score:
                    best_score, best_pos = s, flexed
    if best_pos is None:
        return mol, None, 0.0, mx
    if verbose:
        print(f"{tag} best score {best_score:.3f}/{mx:.3f} ({best_score/mx*100:.1f}%).")
    return mol, best_pos, best_score, mx


def build_output_mol(mol, positions):
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos.tolist())
    out = Chem.RWMol(mol)
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out.GetMol()


def _dock_worker(args):
    RDLogger.DisableLog("rdApp.*")
    name, smiles, sites, excl_vols = args
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mol, positions, best_score, max_score = dock(smiles, sites, excl_vols, label=name, verbose=True)
    except Exception as e:
        buf.write(f"  ERROR: {e}\n")
        return name, None, None, 0.0, 0.0, buf.getvalue()
    mol_block = Chem.MolToMolBlock(mol) if positions is not None else None
    pos_list = positions.tolist() if positions is not None else None
    return name, mol_block, pos_list, best_score, max_score, buf.getvalue()


def main(force_serial=False):
    with open(INPUT_PATH) as fh:
        targets = json.load(fh)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total = len(targets)
    task_args = [
        (name, t["smiles"], t["interaction_sites"], t.get("excluded_volumes", []))
        for name, t in targets.items()
    ]
    results_by_name = {}
    scores_by_name = {}
    workers = min(total, os.cpu_count() or total)

    if force_serial:
        print(f"Running {total} targets serially...", flush=True)
        for idx, (name, smiles, sites, excl_vols) in enumerate(task_args, 1):
            print(f"\n[{idx}/{total}] {name}", flush=True)
            mol, positions, best_score, max_score = dock(smiles, sites, excl_vols, label=name, verbose=True)
            mol_block = Chem.MolToMolBlock(mol) if positions is not None else None
            pos_list = positions.tolist() if positions is not None else None
            results_by_name[name] = (mol_block, pos_list)
            scores_by_name[name] = (best_score, max_score)
    else:
        try:
            print(f"Running {total} targets across {workers} workers...", flush=True)
            ctx = mp.get_context("fork")
            done = 0
            with ctx.Pool(processes=workers) as pool:
                for name, mol_block, pos_list, best_score, max_score, log in pool.imap_unordered(_dock_worker, task_args):
                    done += 1
                    print(f"\n[{done}/{total}] {name}", flush=True)
                    print(log, end="", flush=True)
                    results_by_name[name] = (mol_block, pos_list)
                    scores_by_name[name] = (best_score, max_score)
        except (OSError, RuntimeError) as e:
            print(f"Parallel failed ({e}), falling back to serial...", flush=True)
            for idx, (name, smiles, sites, excl_vols) in enumerate(task_args, 1):
                print(f"\n[{idx}/{total}] {name}", flush=True)
                mol, positions, best_score, max_score = dock(smiles, sites, excl_vols, label=name, verbose=True)
                mol_block = Chem.MolToMolBlock(mol) if positions is not None else None
                pos_list = positions.tolist() if positions is not None else None
                results_by_name[name] = (mol_block, pos_list)
                scores_by_name[name] = (best_score, max_score)

    written = 0
    writer = Chem.SDWriter(OUTPUT_PATH)
    for name in targets:
        result = results_by_name.get(name)
        if result is None:
            continue
        mol_block, pos_list = result
        if pos_list is None:
            continue
        mol = Chem.MolFromMolBlock(mol_block)
        positions = np.array(pos_list)
        out_mol = build_output_mol(mol, positions)
        out_mol.SetProp("_Name", name)
        writer.write(out_mol)
        written += 1
    writer.close()

    _print_summary(targets, scores_by_name, written)


def _print_summary(targets, scores_by_name, written):
    total = len(targets)
    col = max(len(n) for n in targets)

    print("\nresults:", flush=True)
    total_score = total_max = 0.0
    for name in targets:
        best, mx = scores_by_name.get(name, (0.0, 0.0))
        pct = best / mx * 100 if mx else 0.0
        total_score += best
        total_max += mx
        print(f"  {name:<{col}}  {best:.3f} / {mx:.3f}  ({pct:.1f}%)")

    overall = total_score / total_max * 100 if total_max else 0.0
    print(f"  {'overall':<{col}}  {total_score:.3f} / {total_max:.3f}  ({overall:.1f}%)")
    print(f"\nwrote {written}/{total} poses to {OUTPUT_PATH}", flush=True)
