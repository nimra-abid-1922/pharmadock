import math
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from core import (
    score_pose,
    clash_amount,
    steric_clash,
    kabsch,
    pharmacophore_atoms,
    diverse_top_k,
    build_output_mol,
    embed_conformers,
)
from config import CLASH_TOL


def make_sites(*families_weights_positions):
    return [
        {"family": fam, "weight": w, "x": x, "y": y, "z": z}
        for fam, w, (x, y, z) in families_weights_positions
    ]


def test_score_pose_exact_match():
    sites = make_sites(("Acceptor", 1.5, (0.0, 0.0, 0.0)))
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    score = score_pose(atom_pos, sites, {"Acceptor": [0]})
    assert abs(score - 1.5) < 1e-6


def test_score_pose_no_matching_family():
    sites = make_sites(("Donor", 1.0, (0.0, 0.0, 0.0)))
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    score = score_pose(atom_pos, sites, {"Acceptor": [0]})
    assert score == 0.0


def test_score_pose_falls_off_with_distance():
    sites = make_sites(("Acceptor", 1.0, (0.0, 0.0, 0.0)))
    groups = {"Acceptor": [0]}
    close = score_pose(np.array([[0.5, 0.0, 0.0]]), sites, groups)
    far = score_pose(np.array([[3.0, 0.0, 0.0]]), sites, groups)
    assert close > far


def test_score_pose_picks_nearest_atom():
    sites = make_sites(("Acceptor", 1.0, (0.0, 0.0, 0.0)))
    atom_pos = np.array([[5.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    score = score_pose(atom_pos, sites, {"Acceptor": [0, 1]})
    expected = math.exp(-(0.1 / 1.25) ** 2)
    assert abs(score - expected) < 1e-6


def test_clash_amount_no_clash():
    atom_pos = np.array([[5.0, 0.0, 0.0]])
    excl_vols = [{"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.2}]
    assert clash_amount(atom_pos, excl_vols) == 0.0


def test_clash_amount_with_clash():
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    excl_vols = [{"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.2}]
    assert clash_amount(atom_pos, excl_vols) > 0.0


def test_clash_amount_empty_vols():
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    assert clash_amount(atom_pos, []) == 0.0


def test_steric_clash_boundary_safe():
    excl_vols = [{"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.2}]
    safe_dist = 1.2 - CLASH_TOL + 0.01
    atom_pos = np.array([[safe_dist, 0.0, 0.0]])
    assert not steric_clash(atom_pos, excl_vols)


def test_steric_clash_boundary_inside():
    excl_vols = [{"x": 0.0, "y": 0.0, "z": 0.0, "radius": 1.2}]
    inside_dist = 1.2 - CLASH_TOL - 0.01
    atom_pos = np.array([[inside_dist, 0.0, 0.0]])
    assert steric_clash(atom_pos, excl_vols)


def test_kabsch_identity():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R, t = kabsch(pts, pts)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(t, np.zeros(3), atol=1e-6)


def test_kabsch_pure_translation():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([3.0, 1.0, 2.0])
    R, t = kabsch(pts, pts + shift)
    aligned = pts @ R.T + t
    assert np.allclose(aligned, pts + shift, atol=1e-6)


def test_kabsch_no_reflection():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((5, 3))
    R, _ = kabsch(pts, pts)
    assert np.linalg.det(R) > 0


def test_pharmacophore_atoms_has_acceptor_ibuprofen():
    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(O)=O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    groups = pharmacophore_atoms(mol)
    assert len(groups["Acceptor"]) > 0
    assert len(groups["Aromatic"]) > 0


def test_pharmacophore_atoms_caffeine_no_donor():
    mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    groups = pharmacophore_atoms(mol)
    assert len(groups["Donor"]) == 0


def test_diverse_top_k_returns_k():
    rng = np.random.default_rng(1)
    pool = [(float(i), rng.standard_normal((10, 3)) * 10) for i in range(20)]
    result = diverse_top_k(pool, 5)
    assert len(result) == 5


def test_diverse_top_k_pool_smaller_than_k():
    pool = [(1.0, np.zeros((5, 3))), (2.0, np.ones((5, 3)))]
    result = diverse_top_k(pool, 10)
    assert len(result) == 2


def test_build_output_mol_preserves_atom_count():
    mol = Chem.MolFromSmiles("CCO")
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    out = build_output_mol(mol, positions)
    assert out.GetNumAtoms() == n
    assert out.GetNumConformers() == 1


def test_build_output_mol_positions_stored():
    mol = Chem.MolFromSmiles("CCO")
    n = mol.GetNumAtoms()
    positions = np.eye(n, 3)
    out = build_output_mol(mol, positions)
    conf_pos = np.array(out.GetConformer().GetPositions())
    assert np.allclose(conf_pos, positions, atol=1e-4)


def test_embed_conformers_returns_conformers():
    mol = embed_conformers("CCO")
    assert mol.GetNumConformers() >= 1


def test_embed_conformers_rigid_molecule():
    mol = embed_conformers("c1ccccc1")
    assert mol.GetNumConformers() >= 1
