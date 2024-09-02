import argparse
import json
from datetime import datetime
from pprint import pprint

import numpy as np
import wandb
from constants import OPTIONS
from tqdm import tqdm

from utils import cross_entropy_loss, softmax


def analyze_results(result_path):
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
    probs = np.exp(logprobs)
    n_sample_loss = cross_entropy_loss(y_true, logprobs)
    metadata["n_sample_loss"] = n_sample_loss.tolist()
    metadata["task_wise_accuracy"] = (y_true == y_pred).mean(axis=1).tolist()
    metadata["cross_entropy_loss"] = n_sample_loss.mean()
    metadata["comparison"] = {
        "logprobs_vs_y_true": (np.argmax(logprobs, axis=2) == y_true).mean(),
        "y_pred_vs_y_true": (y_pred == y_true).mean(),
        "logprobs_vs_y_pred": (np.argmax(logprobs, axis=2) == y_pred).mean(),
    }
    pprint(metadata)
    with open(f"{result_path}/metadata.json", "w") as f:
        json.dump(metadata, f)
    print("Logging to wandb...")
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    params = {
        "model": metadata["model_name"],
        "dataset.name": "arc_symbolic",
        "dataset.n_tasks": metadata["n_tasks"],
        "dataset.n_samples": metadata["n_samples"],
        "seed": metadata["seed"],
        "prompt_type": metadata["prompt_type"],
    }

    wandb.init(
        entity="dhanya-shridar",
        project="Prequential-ICL",
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

    for i in range(1, n_sample_loss.shape[0]):
        wandb.log({"n_samples": i, "llm/n_sample_loss_nexttoken": n_sample_loss[i]})
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Symbolic LLM")
    parser.add_argument("--result_folder", "-r", type=str, required=True)
    args = parser.parse_args()
    analyze_results(args.result_folder)
