from pathlib import Path
from environ import Env

# Project and source files
DEFAULT_PROJECT_PATH = Path(__file__).parent.parent.parent

# Allow overriding the project path with an environment variable
ENV = Env()
BASE_PATH = Path(ENV("API_DATA_FOLDER", default=DEFAULT_PROJECT_PATH))

PROJECT_PATH = BASE_PATH / "dticlustering"
# DATA_PATH = BASE_PATH / "shared"

# DATASET_SCRATCH = Path('/lustre/fsn1/projects/rech/wgc/uze77wm')
# DATASETS_PATH = DATASET_SCRATCH / 'datasets'
# RUNS_PATH = DATASET_SCRATCH / 'runs'

CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
RUNS_PATH = PROJECT_PATH / 'runs'
RESULTS_PATH = PROJECT_PATH / 'results' # unused?
