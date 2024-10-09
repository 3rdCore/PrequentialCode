python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer_implicit \
    ++logger.tags=[experiments/icl_architectures/regression]
