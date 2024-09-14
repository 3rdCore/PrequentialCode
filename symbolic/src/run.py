import argparse
import os
from time import time

from dataset import get_dataset
from generate import generate_prompts
from process import process_results
from query import Query
from validate import validate_input
from viz import log_results

from utils import save_data, save_prompts, save_results, str2bool


def run(
    dataset_type: str,
    model_name: str,
    n_tasks: int,
    n_samples: int,
    seed: int,
    result_path: str,
    with_options: bool = False,
    **kwargs,
):
    result_folder = f"run_{dataset_type}_{model_name}_{seed}_{int(time())}"
    validate_input(dataset_type, model_name, n_tasks, n_samples, seed, result_path, result_folder)
    print("run_id:", result_folder)
    kwargs.update({"code_length": 4, "num_colours": 6})
    metadata = {
        "model": model_name,
        "dataset.name": dataset_type,
        "dataset.n_tasks": n_tasks,
        "dataset.n_samples": n_samples,
        "dataset.code_length": kwargs["code_length"],
        "dataset.num_colours": kwargs["num_colours"],
        "seed": seed,
        "prompt_type": "no_options" if not with_options else "with_options",
        "run_id": result_folder,
    }
    result_path = os.path.join(result_path, result_folder)
    dataset = get_dataset(dataset_type, n_tasks, n_samples, **kwargs)
    data = dataset.sample(seed)
    save_data(metadata, data, result_path)
    system, prompts = generate_prompts(dataset_type, data, with_options)
    save_prompts(system, prompts, result_path)
    results, query_stats = Query(model_name, dataset_type).query_prompts(system, prompts)
    print(
        "Query stats: success(no retry) - {0}, success(with retry) - {1}, failure - {2}".format(*query_stats)
    )
    metrics, y_pred, logprobs, probs = process_results(results, data)
    save_results(metrics, y_pred, logprobs, probs, results, result_path)
    log_results(metrics, params=metadata)
    print("End of run!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM - Run")
    parser.add_argument("--dataset_type", "-d", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--n_tasks", "-t", type=int, required=True)
    parser.add_argument("--n_samples", "-n", type=int, required=False)
    parser.add_argument("--seed", "-s", type=int, required=False, default=0)
    parser.add_argument("--result_path", "-r", type=str, required=True)
    parser.add_argument("--with_options", "-o", type=str2bool, required=False, default=False)

    args = parser.parse_args()
    dataset_type = args.dataset_type
    model_name = args.model_name
    n_tasks = args.n_tasks
    n_samples = args.n_samples
    seed = args.seed
    result_path = args.result_path
    with_options = args.with_options
    print(
        {
            "Dataset type:": dataset_type,
            "Model name:": model_name,
            "Number of tasks:": n_tasks,
            "Number of samples:": n_samples,
            "Seed:": seed,
            "Result path:": result_path,
            "Options:": with_options,
        }
    )
    run(dataset_type, model_name, n_tasks, n_samples, seed, result_path, with_options)
