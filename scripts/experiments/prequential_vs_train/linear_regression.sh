python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/linear \
    task=meta_optimizer \
    ++task.meta_objective=prequential,train \
    ++dataset.train_dataset.x_dim=3 \
    ++dataset.train_dataset.noise=0.2 \
    ++logger.tags=[experiments/prequential_vs_train/regression]
