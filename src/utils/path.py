from pathlib import Path
from environ import Env

# Project and source files
DEFAULT_PROJECT_PATH = Path(__file__).parent.parent.parent

# Allow overriding the project path with an environment variable
ENV = Env()
PROJECT_PATH = Path(ENV("API_DATA_FOLDER", default=DEFAULT_PROJECT_PATH)) / "dticlustering"

CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
RUNS_PATH = PROJECT_PATH / 'runs'
RESULTS_PATH = PROJECT_PATH / 'results' # unused?
