python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0 \
    dataset=regression/fourier \
    task=fourier_regression \
    ++task.meta_objective=prequential,train \
    ++logger.tags=[experiments/prequential_vs_train/regression]
