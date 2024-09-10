import argparse

import numpy as np


def cross_entropy_loss(y_true, logprobs, keepdims=False):
    y_true_flat = y_true.flatten()
    logprobs_flat = logprobs.reshape(-1, logprobs.shape[-1])
    entropy = logprobs_flat[np.arange(len(y_true_flat)), y_true_flat]
    entropy = -entropy.reshape(y_true.shape)
    return entropy if keepdims else (np.sum(entropy, axis=0) / len(y_true))


def softmax(x):
    if type(x) != np.ndarray:
        x = np.array(x)
    return np.log(np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
