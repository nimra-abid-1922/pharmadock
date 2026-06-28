# Pharmacophore Docking

## How to run

```bash
pip3 install -r requirements.txt
python3 dock.py
```

To run the tests:

```bash
python3 -m pytest test_core.py -v
```

Add `--serial` if multiprocessing causes issues. Output lands at `results/docked_poses.sdf` locally, or `/root/results/docked_poses.sdf` on the eval server, no manual path config required.

---

## What I did

The task was to dock each ligand onto its pharmacophore sites and pick the best pose. My approach was to first cast a wide net of starting positions first, then narrow down step by step through refinement.

For each molecule, I generate a big pool of 3D conformers, then identify which atoms qualify as donors, acceptors, hydrophobes, or aromatics. I use those atoms to build starting poses either by finding three atoms whose spacing matches the site layout, by sitting an atom directly on a site and rotating the rest of the molecule around it, or by embedding a conformer with the highest weight sites already locked in place.

Each pose gets a score based on how close the right atoms land near their sites. The fall off is Gaussian so being a bit off still counts, just less. Any pose that clips an excluded volume gets thrown out.

The best scoring poses then go through two rounds of refinement. First a rigid pass, just moving and rotating the whole molecule until the score peaks. Then for flexible molecules I also let the bonds rotate, using basin hopping to avoid getting stuck. Whatever survives that with no clashes goes into the output SDF.

The rendered poses with pharmacophore sites and exclusion volumes overlaid are in `visualizations/` folder. To run the notebook, open `visualizations/visualize.ipynb` in Google Colab, run cells top to bottom, upload targets.json and docked_poses.sdf when prompted, and the 3D poses will render.
