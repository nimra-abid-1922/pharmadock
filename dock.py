import argparse

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from core import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharmacophore-based ligand docking")
    parser.add_argument("--serial", action="store_true",help="disable multiprocessing (easier to debug)")
    args = parser.parse_args()
    main(force_serial=args.serial)
