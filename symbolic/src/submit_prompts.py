import argparse
import json
import os
from datetime import datetime
from pprint import pprint
from time import time

import numpy as np
import wandb
from generate import generate_prompt, generate_prompt_v2
from llms import GPT, Model
from xarc import TASK_LIST
from xarc import generate_data_with_options as xarc_generate

from utils import str2bool


def submit_prompts(
    model_name, n_tasks, n_samples, save_folder, seed, tasks=TASK_LIST, random=False, with_options=False
):
    if random:
        np.random.seed(seed)
        tasks = np.random.choice(TASK_LIST, n_tasks).tolist()
    else:
        n_tasks = len(tasks)

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
        "tasks": tasks,
        "n_samples": n_samples,
        "n_tasks": n_tasks,
        "seed": seed,
        "model_name": model.value,
        "prompt_type": "with_options" if with_options else "no_options",
    }
    print("Saving metadata...")
    with open(os.path.join(result_path, "metadata.json"), "w+") as f:
        json.dump(meta_data, f)
    i = 0
    for task in xarc_generate(tasks=tasks, num_samples=n_samples, seed=seed):
        prompts = generate_prompt(task, with_options=with_options)
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


def submit_prompts_v2(
    model_name,
    n_tasks,
    n_samples,
    n_queries,
    save_folder,
    seed,
    tasks=TASK_LIST,
    random=False,
    with_options=False,
):
    if random:
        np.random.seed(seed)
        tasks = np.random.choice(TASK_LIST, n_tasks).tolist()
    else:
        n_tasks = len(tasks)

    model = Model._value2member_map_[model_name]
    llm = GPT(model=model, temperature=0, top_logprobs=20, max_tokens=20)
    logprob_tasks = []
    y_true_tasks = []
    y_pred_tasks = []

    if not os.path.isdir(save_folder):
        raise FileExistsError(f"File already exists: {save_folder}")
    result_folder = f"results_task={tasks[0]}_{int(time())}"
    result_path = os.path.join(save_folder, result_folder)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    meta_data = {
        "tasks": tasks * n_queries,
        "n_tasks": n_tasks * n_queries,
        "n_samples": n_samples,
        "n_queries": n_queries,
        "seed": seed,
        "model_name": model.value,
        "prompt_type": "with_options" if with_options else "no_options",
    }
    print("Saving metadata...")
    with open(os.path.join(result_path, "metadata.json"), "w+") as f:
        json.dump(meta_data, f)
    i = 0
    for task in xarc_generate(tasks=tasks, num_samples=n_samples + n_queries, seed=seed):
        prompts = generate_prompt_v2(task, n_samples, n_queries, with_options=with_options)
        results, query_log = llm(prompts)
        y_true_tasks = list(zip(*task))[3][-5:]
        for i in range(len(results[0])):
            y_pred = [int(r[i]["answer"]) for r in results]
            logprobs = [r[i]["logprobs"] for r in results]
            y_pred_tasks.append(y_pred)
            logprob_tasks.append(logprobs)
        i += 1
    y_true = np.array(y_true_tasks)[:, None].repeat(n_samples, axis=1)
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
    parser.add_argument("--n_queries", "-q", type=int, required=False)
    parser.add_argument("--seed", "-s", type=int, required=False, default=0)
    parser.add_argument("--save_folder", "-r", type=str, required=True)
    parser.add_argument("--with_options", "-o", type=str2bool, required=False, default=False)

    args = parser.parse_args()
    model_name = args.model_name
    n_tasks = args.n_tasks
    n_samples = args.n_samples
    n_queries = args.n_queries
    seed = args.seed
    save_folder = args.save_folder
    with_options = args.with_options
    print(
        {
            "Model name:": model_name,
            "Number of tasks:": n_tasks,
            "Number of samples:": n_samples,
            "Number of queries:": n_queries,
            "Seed:": seed,
            "Save folder path:": save_folder,
            "Options:": with_options,
        }
    )

    # submit_prompts(
    #     model_name,
    #     n_tasks,
    #     n_samples,
    #     save_folder=save_folder,
    #     seed=seed,
    #     random=False,
    #     with_options=with_options,
    # )
    submit_prompts_v2(
        model_name,
        n_tasks,
        n_samples,
        n_queries,
        save_folder=save_folder,
        seed=seed,
        random=False,
        tasks=["x1"],
        with_options=with_options,
    )
