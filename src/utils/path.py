from pathlib import Path
from environ import Env

# Project and source files
DEFAULT_PROJECT_PATH = Path(__file__).parent.parent.parent

# Allow overriding the project path with an environment variable
ENV = Env()
PROJECT_PATH = Path(ENV("DTI_DATA_FOLDER", default=DEFAULT_PROJECT_PATH))

DATASET_SCRATCH = Path('/lustre/fsn1/projects/rech/wgc/uze77wm')

CONFIGS_PATH = PROJECT_PATH / 'configs'
#DATASETS_PATH = DATASET_SCRATCH / 'datasets'
DATASETS_PATH = PROJECT_PATH / 'datasets'
#RUNS_PATH = DATASET_SCRATCH / 'runs' # PROJECT_PATH / 'runs'
RUNS_PATH = PROJECT_PATH / 'runs'
RESULTS_PATH = PROJECT_PATH / 'results' # unused?
