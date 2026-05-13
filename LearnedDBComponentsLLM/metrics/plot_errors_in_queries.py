from metrics.plotting import plot_query_error_overview
from utils.session_utils import get_latest_json_path
from utils.io_utils import read_json_file
import os
from pathlib import Path

OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "output"))
latest_json = get_latest_json_path(OUTPUT_FOLDER)
plot_dir = latest_json.parent / "plots"
queries = read_json_file(latest_json)

plot_query_error_overview(queries, plot_dir)