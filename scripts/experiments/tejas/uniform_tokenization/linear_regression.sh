python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/linear \
    task=meta_optimizer \
    ++task.meta_objective=prequential \
    ++dataset.train_dataset.x_dim=3 \
    ++dataset.train_dataset.noise=0.2 \
    task/context_aggregator=transformer2 \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/separate_tokenization]

python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/linear \
    task=meta_optimizer_implicit \
    ++dataset.train_dataset.x_dim=3 \
    ++dataset.train_dataset.noise=0.2 \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/separate_tokenization]
