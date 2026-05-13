import sys
import os
from pathlib import Path


from dotenv import load_dotenv
from metrics.validating import validty_pipeline
from metrics.SQL_Complexity import run_complexity_pipeline
from metrics.selective_non_selective import run_selective_non_selective_pipeline
from metrics.calculate_rows import calculate_rows_pipeline

load_dotenv(root / ".env")

session_name = sys.argv[1] if len(sys.argv) > 1 else None

# Some pipelines might not take session_name yet, but we will pass it to complexity at least.
try:
    validty_pipeline(session_name)
except TypeError:
    validty_pipeline()

try:
    run_complexity_pipeline(recompute=True, session_name=session_name)
except TypeError:
    run_complexity_pipeline(recompute=True)

try:
    run_selective_non_selective_pipeline(session_name)
except TypeError:
    run_selective_non_selective_pipeline()