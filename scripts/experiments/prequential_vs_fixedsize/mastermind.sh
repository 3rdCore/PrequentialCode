FIXED_SIZES=(3 9 30)

for FIXED_SIZE in ${FIXED_SIZES[@]}; do
    python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs task=meta_optimizer_symbolic ++task.meta_objective=prequential ++task.context_aggregator.x_dim=50 ++predictor.x_dim=48 ++predictor.y_dim=18 \
    ++logger.tags=[experiments/prequential_vs_fixedsize/symbolic] \
    seed=0,1,2,3,4 \
    dataset=symbolic/mastermind \
    ++dataset.train_dataset.n_samples=$(($FIXED_SIZE + 1)) \
    ++task.min_train_samples=$FIXED_SIZE
done
