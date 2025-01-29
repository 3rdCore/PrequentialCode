python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=symbolic/hmm_supervised \
    task=meta_optimizer_symbolic \
    ++task.meta_objective=prequential,train \
    ++context_aggregator.x_dim=306 \
    ++predictor.x_dim=256 \
    ++predictor.y_dim=50 \
    ++trainer.max_epochs=80 \
    ++logger.tags=[experiments/prequential_vs_train/symbolic]
