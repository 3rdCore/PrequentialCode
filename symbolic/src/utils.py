import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch


def batched_bincount(x: torch.LongTensor, max_val: int) -> torch.LongTensor:
    cnt = x.new_zeros(*x.shape[:-1], max_val)
    return cnt.scatter_add(dim=-1, index=x, src=x.new_ones(()).expand_as(x))


def softmax(x):
    if type(x) != np.ndarray:
        x = np.array(x)
    return np.log(np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True))


def cross_entropy_loss(y_true, logprobs, keepdims=False):
    y_true_flat = y_true.flatten()
    logprobs_flat = logprobs.reshape(-1, logprobs.shape[-1])
    entropy = logprobs_flat[np.arange(len(y_true_flat)), y_true_flat]
    entropy = -entropy.reshape(y_true.shape)
    return entropy if keepdims else (np.sum(entropy, axis=0) / len(y_true))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_results(result_path):
    result_path = os.path.join(result_path, "results")
    with open(os.path.join(result_path, "metrics.json"), "r") as f:
        metrics = json.load(f)
    return metrics


def load_metadata(result_path):
    with open(os.path.join(result_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return metadata


def save_data(metadata, data, result_path):
    with open(os.path.join(result_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    data_path = os.path.join(result_path, "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    x, y = data
    with open(os.path.join(data_path, "x.npy"), "wb") as f:
        np.save(f, x)
    with open(os.path.join(data_path, "y_true.npy"), "wb") as f:
        np.save(f, y)
    print(f"Data is saved in '{data_path}'")
    return data_path


def save_prompts(system: str, prompts: pd.DataFrame, result_path):
    with open(os.path.join(result_path, "system.txt"), "w") as f:
        f.write(system)
    prompts_path = os.path.join(result_path, "prompts")
    if not os.path.exists(prompts_path):
        os.mkdir(prompts_path)
    for t in range(len(prompts)):
        prompt_path = os.path.join(prompts_path, f"prompts_{t}.csv")
        prompts[t].to_csv(prompt_path, sep="|")
    print(f"Prompts are saved in '{prompts_path}'")
    return prompts_path


def save_results(metrics, y_pred, logprobs, probs, results, result_path):
    result_path = os.path.join(result_path, "results")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    with open(os.path.join(result_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(result_path, "y_pred.npy"), "wb") as f:
        np.save(f, y_pred)
    with open(os.path.join(result_path, "logprobs.npy"), "wb") as f:
        np.save(f, logprobs)
    with open(os.path.join(result_path, "probs.npy"), "wb") as f:
        np.save(f, probs)
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"Results are saved in '{result_path}'")


def extract_html_tags(text, keys):
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        # pattern = r"\b(?:[0-4])\b"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict
