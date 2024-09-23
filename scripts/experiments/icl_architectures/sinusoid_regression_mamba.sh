MAMBAVERSION=$1

python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer \
    ++task.meta_objective=prequential \
    task/context_aggregator=mamba \
    ++task.context_aggregator.mixer_type=Mamba${MAMBAVERSION} \
    ++logger.tags=[experiments/icl_architectures/regression]
