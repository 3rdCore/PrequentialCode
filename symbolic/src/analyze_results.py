import argparse
import json
import os
import re
from datetime import datetime
from pprint import pprint

import numpy as np
import wandb
from constants import OPTIONS
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import cross_entropy_loss, softmax, str2bool


def analyze_results(result_path, avg_across_tasks):
    run_id = re.search(r"_([0-9]+)", result_path).group(0)
    logprobs_json = json.load(open(f"{result_path}/logprobs.json"))
    y_true = np.load(f"{result_path}/y_true.npy")
    y_pred = np.load(f"{result_path}/y_pred.npy")
    metadata = json.load(open(f"{result_path}/metadata.json"))
    logprobs = []
    for task in tqdm(logprobs_json, desc="Tasks:"):
        logprob = []
        for sample in task:
            d_value = min(sample.values())
            logprob.append([sample.get(o, d_value) for o in OPTIONS])
        logprobs.append(logprob)
    logprobs = np.array(logprobs)
    logprobs = softmax(logprobs)
    n_sample_loss, probs, probs_rand, results = compute_metrics(y_true, y_pred, logprobs, avg_across_tasks)
    pprint(results)
    with open(f"{result_path}/results.json", "w") as f:
        json.dump(results, f)
    params = {
        "model": metadata["model_name"],
        "dataset.name": "arc_symbolic",
        "dataset.n_tasks": metadata["n_tasks"],
        "dataset.n_samples": metadata["n_samples"],
        "seed": metadata["seed"],
        "prompt_type": metadata["prompt_type"],
        "run_id": f"run_{run_id}",
    }
    tasks = None if avg_across_tasks else metadata["tasks"]
    print(avg_across_tasks)
    log_results(y_true, n_sample_loss, probs, probs_rand, params, tasks=tasks)
    plot_results(y_true, y_pred, logprobs, probs, result_path)
    pprint(results)
    print("Done!")


def compute_metrics(y_true, y_pred, logprobs, avg_across_tasks=False):
    print("Computing metrics...")
    print("y_true:", y_true.shape, "y_pred", y_pred.shape, "logprobs", logprobs.shape)
    print(y_true)
    n_sample_loss = cross_entropy_loss(y_true, logprobs, not avg_across_tasks)
    probs = np.exp(logprobs)
    probs_rand = np.random.rand(*probs.shape)
    probs_rand = (
        probs_rand.reshape(-1, 5)[np.arange(y_true.size), y_true.flatten()]
        .reshape(probs_rand.shape[:-1])
        .mean(axis=0)
    )
    probs = probs.reshape(-1, 5)[np.arange(y_true.size), y_true.flatten()].reshape(probs.shape[:-1])

    results = {
        "n_sample_loss": n_sample_loss.tolist(),
        "task_wise_accuracy": (y_true == y_pred).mean(axis=1).tolist(),
        "cross_entropy_loss": n_sample_loss.mean(),
        "comparison": {
            "logprobs_vs_y_true": (np.argmax(logprobs, axis=2) == y_true).mean(),
            "y_pred_vs_y_true": (y_pred == y_true).mean(),
            "logprobs_vs_y_pred": (np.argmax(logprobs, axis=2) == y_pred).mean(),
        },
    }
    return n_sample_loss, probs, probs_rand, results


def plot_results(y_true, y_pred, n_sample_loss, probs, results_path):
    print(np.mean(probs, axis=0).shape)
    plt.errorbar(
        x=np.arange(0, probs.shape[1]),
        y=np.mean(probs, axis=0),
        yerr=np.std(probs, axis=0) / np.sqrt(probs.shape[0]),
        capsize=5,
        color="skyblue",
        ecolor="black",
    )
    plt.xlabel("Sample ")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(results_path, "probs.png"))
    plt.show()

    plt.errorbar(
        x=np.arange(0, n_sample_loss.shape[1]),
        y=np.mean(n_sample_loss, axis=0),
        yerr=np.std(n_sample_loss, axis=0) / np.sqrt(n_sample_loss.shape[0]),
        capsize=5,
        color="skyblue",
        ecolor="black",
    )
    plt.xlabel("Sample ")
    plt.ylabel("n_sample_loss")
    plt.savefig(os.path.join(results_path, "n_sample_loss.png"))
    plt.show()

    plt.bar(x=np.arange(0, y_true.shape[1]), y=np.mean(y_true == y_pred, axis=0), capsize=5, color="skyblue")
    plt.xlabel("Sample ")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(results_path, "accuracy.png"))
    plt.show()


def log_results(y_true, n_sample_loss, probs, probs_rand, params, tasks=None):
    print("Logging to wandb...")
    if tasks == None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            entity="dhanya-shridar",
            project="Prequential-ICL",
            name=name,
            tags=["SYMBOLIC/pre-trained_llm"],
            config=params,
        )
        for i, (p, p_r) in enumerate(zip(probs.mean(axis=0), probs_rand), start=1):
            wandb.log({"n_samples": i, "llm/p": p, "llm/p_rand": p_r})

        for i in range(1, n_sample_loss.shape[0]):
            wandb.log({"n_samples": i, "llm/n_sample_loss_nexttoken": n_sample_loss[i]})
        wandb.finish()
    else:
        for t, task in enumerate(tasks):
            params["task_name"] = task
            name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wandb.init(
                entity="dhanya-shridar",
                project="Prequential-ICL",
                name=name,
                tags=["SYMBOLIC/pre-trained_llm"],
                config=params,
            )
            for s in range(n_sample_loss.shape[1]):
                wandb.log(
                    {"n_samples": s, "llm/n_sample_loss_nexttoken": n_sample_loss[t, s], "llm/p": probs[t, s]}
                )
            wandb.finish()
        params["task_name"] = "rand_p"
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            entity="dhanya-shridar",
            project="Prequential-ICL",
            name=name,
            tags=["SYMBOLIC/pre-trained_llm"],
            config=params,
        )
        for s in range(probs_rand.size):
            wandb.log({"n_samples": s, "llm/p_rand": probs_rand[s]})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM")
    parser.add_argument("--result_folder", "-r", type=str, required=True)
    parser.add_argument("--avg_across_tasks", "-a", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    analyze_results(args.result_folder, args.avg_across_tasks)
