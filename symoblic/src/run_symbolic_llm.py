import argparse
import json
import os
from datetime import datetime
from pprint import pprint

import numpy as np
import wandb
from generate import generate_prompt
from llms import GPT, Model
from xarc import TASK_LIST
from xarc import generate_data_with_options as xarc_generate

from utils import cross_entropy_loss, softmax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM")
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--n_tasks", "-t", type=int, required=True)
    parser.add_argument("--n_samples", "-n", type=int, required=False)
    parser.add_argument("--seed", "-s", type=int, required=True)
    args = parser.parse_args()
    model_name = args.model_name
    n_tasks = args.n_tasks
    n_samples = args.n_samples
    seed = args.seed
    random = True
    tasks = np.random.choice(TASK_LIST, n_tasks) if random else TASK_LIST
    model = Model._value2member_map_[model_name]
    llm = GPT(model=model, temperature=0, top_logprobs=20, max_tokens=20)
    logprob_tasks = []
    y_true_tasks = []
    y_pred_tasks = []

    for task in xarc_generate(tasks=tasks, num_samples=n_samples, seed=seed):
        prompts = generate_prompt(task, with_options=True)
        results, query_log = llm(prompts)
        y_true = list(zip(*task))[3]
        y_pred = [int(r["answer"]) for r in results]
        logprobs = [r["logprobs"] for r in results]
        y_true_tasks.append(y_true)
        y_pred_tasks.append(y_pred)
        logprob_tasks.append(logprobs)

    y_true = np.array(y_true_tasks)
    y_pred = np.array(y_pred_tasks)
    logprobs = softmax(logprob_tasks)
    probs = np.exp(logprobs)
    n_sample_loss = cross_entropy_loss(y_true, logprobs)

    metadata = dict()
    metadata["tasks"] = tasks.tolist()
    metadata["n_sample_loss"] = n_sample_loss.tolist()
    metadata["task_wise_accuracy"] = (y_true == y_pred).mean(axis=1).tolist()
    metadata["cross_entropy_loss"] = n_sample_loss.mean()
    metadata["comparison"] = {
        "logprobs_vs_y_true": (np.argmax(logprobs, axis=2) == y_true).mean(),
        "y_pred_vs_y_true": (y_pred == y_true).mean(),
        "logprobs_vs_y_pred": (np.argmax(logprobs, axis=2) == y_pred).mean(),
    }

    print("Logging to wandb...")
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    params = {
        "model": model_name,
        "dataset.name": "arc_symbolic",
        "dataset.n_tasks": n_tasks,
        "dataset.n_samples": n_samples,
    }
    metadata["params"] = params

    pprint(metadata)
    store_path = f"../experiments/results/{name}"
    os.mkdir(store_path)
    np.save(f"{store_path}/y_true.npy", y_true)
    np.save(f"{store_path}/y_pred.npy", y_pred)
    np.save(f"{store_path}/logprobs.npy", logprobs)
    with open(f"{store_path}/metadata.json", "w") as f:
        json.dump(metadata, f)

    wandb.init(
        project="Prequential-ICL",
        entity="dhanya-shridar",
        name=name,
        tags=["SYMBOLIC/pre-trained_llm"],
        config=params,
    )
    probs_flatten = probs.reshape(-1, 5)[np.arange(y_true.size), y_true.flatten()].reshape(probs.shape[:-1])
    probs_rand = np.random.rand(*probs.shape)
    probs_rand_flatten = probs_rand.reshape(-1, 5)[np.arange(y_true.size), y_true.flatten()].reshape(
        probs_rand.shape[:-1]
    )

    for i, (p, p_r) in enumerate(zip(probs_flatten.mean(axis=0), probs_rand_flatten.mean(axis=0)), start=1):
        wandb.log({"n_samples": i, "llm/p": p, "llm/p_rand": p_r})

    # for i in range(1, probs.shape[1]):
    #     for j in range(probs.shape[0]):
    #         wandb.log({"n_samples": i, "n_tasks" : j, "llm/p": probs_flatten[j, i]})

    for i in range(1, n_sample_loss.shape[0]):
        wandb.log({"n_samples": i, "llm/n_sample_loss_nexttoken": n_sample_loss[i]})
    wandb.finish()
    print("Done!")
