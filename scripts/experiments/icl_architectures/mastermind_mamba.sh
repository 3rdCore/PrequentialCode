MAMBAVERSION=$1
NSAMPLES=0
GRES=""
[ $MAMBAVERSION = 1 ] && NSAMPLES=1000 || NSAMPLES=800
[ $MAMBAVERSION = 1 ] && GRES="gpu:1" || GRES="gpu:l40s:1"

python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=symbolic/mastermind \
    task=meta_optimizer_symbolic \
    ++task.meta_objective=prequential \
    task/context_aggregator=mamba \
    ++task.context_aggregator.mixer_type=Mamba${MAMBAVERSION} \
    ++task.context_aggregator.x_dim=66 \
    ++context_aggregator.x_dim=66 \
    ++predictor.x_dim=48 \
    ++predictor.y_dim=18 \
    ++trainer.max_epochs=120 \
    ++dataset.train_dataset.n_samples=${NSAMPLES} \
    hydra.launcher.gres=${GRES} \
    ++logger.tags=[experiments/icl_architectures/symbolic]
