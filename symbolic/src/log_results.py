import argparse

import numpy as np
from viz import log_results

from utils import load_metadata, load_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM - Log Results")
    parser.add_argument("--run_path", "-r", type=str, required=True)
    args = parser.parse_args()
    metadata = load_metadata(args.run_path)
    metrics = load_results(args.run_path)
    log_results(metrics, params=metadata)
