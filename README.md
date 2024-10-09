# In-context learning and Occam's razor
This is code for reproducing experiments in the paper [In-Context Learning and Occam's Razor](TODO).

## Dependencies
Install required python packages:
```
pip install -r requirements.txt
```

*Notes*:
- `causal-conv1d` and `mamba-ssm[causal-conv1d]` are only necessary for running experiments with SSM models.
- `submitit` and `hydra-submitit-launcher` are only necessary for running experiments with the `submitit` launcher on a computing cluster.
- `pre-commit` is only necessary for running the pre-commit checks.

We use Weights & Biases for logging experiment data. You can create a free account at [wandb.ai](https://wandb.ai) and run `wandb login` to authenticate your machine.


## Reproducing main experiments

### Experiments 3.1-3.3

Commands for running most of the experiments that appear in the main paper can be found in: `scripts/experiments/`. Specifically:
- `scripts/experiments/prequential_vs_train/`: Experiments section 3.1 of the paper.
- `scripts/experiments/sgd_vs_prequential/`: Experiments section 3.2 of the paper.
- `scripts/experiments/icl_architectures/`: Experiments section 3.3 of the paper.

Note that you will need to change minor details in these scripts (and their associated hydra configurations) to run them on your machine:
- Change `logger.entity` in `configs/train.yaml` to your Weights & Biases username.
- Change `save_dir=[your save directory]` in the run scripts to where you want logs and models to be saved.
- If *not* using `submitit` on a computing cluster, remove `--multirun hydra/launcher=[launcher file]` from the run scripts *and* any time a script argument has a "," in it (which sweeps over an argument's value), remove the list of arguments and run the script multiple times with different values for that argument.
- If using `submitit`, you will need to create a new launcher configuration in `configs/hydra/launcher/` and change the `hydra/launcher=[your launcher file]` argument in the run scripts to match your computing cluster's configuration.

### Experiment 3.4

These experiments require an API key for OpenAI GPT. Afterwards, you can reproduce the experiments using the command in:
- `symbolic/scripts.sh`

### Experiment 3.5

This experiment is not included in this repository, as it uses code for a different project that is not yet published. However, the code for this experiment is available upon request.
