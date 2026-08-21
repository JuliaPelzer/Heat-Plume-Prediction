from pathlib import Path
import sys

# Find the project root by going up two levels from the current file's directory
# and add it to the start of the Python path.
project_root = Path.cwd().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")
print(sys.path)

from preprocessing.preprocessing import import_dataset
from step2_streamlines.streamlines_main import build_streamlines

# path_orig = Path("/scratch/sgs/pelzerja/benchmarking/data/step2")
# path_desti = Path("/scratch/sgs/pelzerja/datasets_prepared/bm/step2")
# import_dataset(path_orig, path_desti, test_data=False)

path_prepared_data = Path("/scratch/sgs/pelzerja/datasets_prepared/bm/step2")
# STEP 2: calculate streamlines with simulated or with predicted velocity fields
build_streamlines(path_prepared_data, method="Radau")