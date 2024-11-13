python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer \
    ++task.meta_objective=prequential \
    task/context_aggregator=transformer2  \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/separate_tokenization]


python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer_implicit \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/separate_tokenization]
