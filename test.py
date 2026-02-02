import os
import subprocess
from pathlib import Path
import BeamsPackage as bp

"""2D Beams Lattice Generator Test file"""

def main():
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_DATA_DIR = BASE_DIR / "cells.json"
    MESH_DATA_DIR = BASE_DIR / "output.mesh"

    # bp.copy_example_to_cwd("cells.json")
    bp.mesher_beams1D(case="cell_kagome_lattice_test", fimport=DEFAULT_DATA_DIR, fexport=BASE_DIR, ui=True)
    bp.parse_beam_inria_mesh_file(fimport=MESH_DATA_DIR, fexport=BASE_DIR)

if __name__ == "__main__":
    main()
