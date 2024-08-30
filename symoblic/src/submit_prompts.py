import argparse
import json
import os
from datetime import datetime
from pprint import pprint
from time import time

import numpy as np
import wandb
from generate import generate_prompt
from llms import GPT, Model
from xarc import TASK_LIST
from xarc import generate_data_with_options as xarc_generate

from utils import cross_entropy_loss, softmax


def submit_prompts(model_name, n_tasks, n_samples, save_folder, seed, random=True):
    tasks = np.random.choice(TASK_LIST, n_tasks) if random else TASK_LIST
    model = Model._value2member_map_[model_name]
    llm = GPT(model=model, temperature=0, top_logprobs=20, max_tokens=20)
    logprob_tasks = []
    y_true_tasks = []
    y_pred_tasks = []

    if not os.path.isdir(save_folder):
        raise FileExistsError(f"File already exists: {save_folder}")
    result_folder = f"results_seed={seed}_{int(time())}"
    result_path = os.path.join(save_folder, result_folder)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    meta_data = {
        "tasks": tasks.tolist(),
        "n_samples": n_samples,
        "n_tasks": n_tasks,
        "seed": seed,
        "model_name": model.value,
        "prompt_type": "no_options",
    }
    print("Saving metadata...")
    with open(os.path.join(result_path, "metadata.json"), "w+") as f:
        json.dump(meta_data, f)
    i = 0
    for task in xarc_generate(tasks=tasks, num_samples=n_samples, seed=seed):
        prompts = generate_prompt(task, with_options=True)
        results, query_log = llm(prompts)
        y_true = list(zip(*task))[3]
        y_pred = [int(r["answer"]) for r in results]
        logprobs = [r["logprobs"] for r in results]
        y_true_tasks.append(y_true)
        y_pred_tasks.append(y_pred)
        logprob_tasks.append(logprobs)
        i += 1
    y_true = np.array(y_true_tasks)
    y_pred = np.array(y_pred_tasks)
    np.save(f"{result_path}/y_true.npy", y_true)
    np.save(f"{result_path}/y_pred.npy", y_pred)
    with open(f"{result_path}/logprobs.json", "w") as f:
        json.dump(logprob_tasks, f)
    print(f"Results are saved in {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM")
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--n_tasks", "-t", type=int, required=True)
    parser.add_argument("--n_samples", "-n", type=int, required=False)
    parser.add_argument("--seed", "-s", type=int, required=True)
    parser.add_argument("--save_folder", "-r", type=str, required=True)

    args = parser.parse_args()
    model_name = args.model_name
    n_tasks = args.n_tasks
    n_samples = args.n_samples
    seed = args.seed
    save_folder = args.save_folder
    submit_prompts(model_name, n_tasks, n_samples, save_folder=save_folder, seed=seed)
