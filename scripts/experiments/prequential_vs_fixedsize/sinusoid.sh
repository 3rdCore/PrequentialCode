FIXED_SIZES=(3 9 20)

for FIXED_SIZE in ${FIXED_SIZES[@]}; do
    python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs task=meta_optimize ++task.meta_objective=prequential \
    ++logger.tags=[experiments/prequential_vs_fixedsize/regression] \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    ++dataset.train_dataset.n_samples=$(($FIXED_SIZE + 1)) \
    ++task.min_train_samples=$FIXED_SIZE
done
