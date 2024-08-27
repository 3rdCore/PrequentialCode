import numpy as np


def cross_entropy_loss(y_true, logprobs):
    y_true_flat = y_true.flatten()
    logprobs_flat = logprobs.reshape(-1, logprobs.shape[-1])
    entropy = logprobs_flat[np.arange(len(y_true_flat)), y_true_flat]
    entropy = entropy.reshape(y_true.shape)
    return -np.sum(entropy, axis=0) / len(y_true)


def softmax(x):
    if type(x) != np.ndarray:
        x = np.array(x)
    return np.log(np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True))
