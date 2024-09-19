python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
    dataset=regression/linear \
    task=sgd_optimizer \
    ++dataset.train_dataset.x_dim=3 \
    ++predictor.n_layers=5 \
    ++predictor.h_dim=64 \
    ++datamodule.batch_size=64 \
    ++dataset.train_dataset.n_samples=2000 \
    ++dataset.train_dataset.noise=0.2 \
    ++task.inner_epochs=2000 \
    ++task.lr=0.0001 \
    ++trainer.max_epochs=2000000 \
    ++callbacks.monitor=val_loss \
    ++callbacks.min_delta=1e-3 \
    ++callbacks.patience=10 \
    ++trainer.gradient_clip_val=0.05 \
    ++logger.tags=[experiments/sgd_vs_prequential/regression]
