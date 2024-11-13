python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4  \
    dataset=symbolic/mastermind   \
    task=meta_optimizer_symbolic  \
    ++task.meta_objective=prequential \
    ++predictor.x_dim=48  \
    ++predictor.y_dim=18  \
    ++trainer.max_epochs=80  \
    task/context_aggregator=transformer2 \
    ++logger.tags=[experiments/icl_architectures/symbolic,experiments/separate_tokenization]

python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=symbolic/mastermind \
    task=meta_optimizer_implicit_symbolic \
    ++task.model.x_dim=48 \
    ++task.model.y_dim=18 \
    ++trainer.max_epochs=80 \
    ++logger.tags=[experiments/icl_architectures/symbolic,experiments/separate_tokenization]
