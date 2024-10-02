import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb


def log_results(metrics, params):
    n_sample_loss = np.array(metrics["n_sample_loss_nexttoken"])
    n_sample_loss_marg = np.array(metrics["n_sample_loss_nexttoken_marg"])
    probs = np.array(metrics["probs"])
    probs_rand = np.array(metrics["probs_rand"])
    probs_marg = np.array(metrics["probs_marg"])
    print("Logging to wandb...")
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="dhanya-shridar",
        project="Prequential-ICL",
        name=name,
        tags=["tejas/llm-symbolic"],
        config=params,
    )
    for i, (p, p_m, p_r, loss, loss_m) in enumerate(
        zip(probs, probs_marg, probs_rand, n_sample_loss, n_sample_loss_marg)
    ):
        wandb.log(
            {
                "n_samples": i + 1,
                "p": p,
                "p_marg": p_m,
                "p_rand": p_r,
                "n_sample_loss_nexttoken": loss,
                "n_sample_loss_nexttoken_marg": loss_m,
            }
        )
    wandb.finish()


def plot_y_histogram(result_paths, code_length, ax):
    y_preds_0 = []
    y_preds_1 = []

    y_trues_0 = []
    y_trues_1 = []
    unique_values = np.arange(0, code_length + 1)
    for result_path in result_paths:
        y_pred = np.load(os.path.join(result_path, "results", "y_pred.npy"))
        y_true = np.load(os.path.join(result_path, "data", "y_true.npy"))
        counts = np.bincount(y_pred[..., 0].flatten(), minlength=code_length + 1)
        y_preds_0.append(counts)
        counts = np.bincount(y_true[..., 0].flatten(), minlength=code_length + 1)
        y_trues_0.append(counts)
        counts = np.bincount(y_pred[..., 1].flatten(), minlength=code_length + 1)
        y_preds_1.append(counts)
        counts = np.bincount(y_true[..., 1].flatten(), minlength=code_length + 1)
        y_trues_1.append(counts)

    y_preds_0 = np.stack(y_preds_0)
    y_trues_0 = np.stack(y_trues_0)
    y_preds_1 = np.stack(y_preds_1)
    y_trues_1 = np.stack(y_trues_1)

    ax.errorbar(
        x=unique_values,
        y=y_preds_0.mean(axis=0),
        yerr=y_preds_0.std(axis=0),
        label="y_pred_exact_match",
        c="red",
    )
    ax.errorbar(
        unique_values,
        y_trues_0.mean(axis=0),
        yerr=y_trues_0.std(axis=0),
        label="y_true_exact_match",
        linestyle="--",
        c="red",
    )
    ax.errorbar(
        x=unique_values,
        y=y_preds_1.mean(axis=0),
        yerr=y_preds_1.std(axis=0),
        label="y_pred_correct_digit",
        c="blue",
    )
    ax.errorbar(
        x=unique_values,
        y=y_trues_1.mean(axis=0),
        yerr=y_trues_1.std(axis=0),
        label="y_true_correcT_digit",
        linestyle="--",
        c="blue",
    )
    ax.set_xticks(unique_values)
    ax.set_ylabel("Freuency")
    ax.set_xlabel("Value")
    ax.set_title(f"Freq Distribution ({code_length})")
