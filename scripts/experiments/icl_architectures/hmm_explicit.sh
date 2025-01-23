python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=symbolic/hmm \
    task=meta_optimizer_symbolic \
    task/context_aggregator=transformer_sequence \
    ++task.context_aggregator.x_dim=50 \
    ++context_aggregator.x_dim=50 \
    ++predictor.x_dim=50 \
    ++predictor.y_dim=50 \
    ++logger.tags=[experiments/icl_architectures/hmm]
