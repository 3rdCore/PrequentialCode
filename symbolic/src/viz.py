from datetime import datetime

import numpy as np
import wandb


def log_results(metrics, params):
    n_sample_loss = np.array(metrics["n_sample_loss_nexttoken"])
    probs = np.array(metrics["probs"])
    probs_rand = np.array(metrics["probs_rand"])
    print("Logging to wandb...")
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        entity="dhanya-shridar",
        project="Prequential-ICL",
        name=name,
        tags=["tejas/llm-symbolic"],
        config=params,
    )
    for i, (p, p_r, loss) in enumerate(zip(probs, probs_rand, n_sample_loss)):
        wandb.log({"n_samples": i + 1, "p": p, "p_rand": p_r, "n_sample_loss_nexttoken": loss})
    wandb.finish()
