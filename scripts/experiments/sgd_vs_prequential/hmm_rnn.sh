python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14  \
    dataset=symbolic/hmm \
    task=sgd_optimizer \
    ++datamodule.val_prop=0.0 \
    ++datamodule.max_train_samples=200 \
    ++task.loss_fn._target_=torch.nn.CrossEntropyLoss \
    ++task.loss_fn.reduction=none \
    ++logger.tags=[experiments/sgd_vs_prequential/symbolic_rnn] \
    task/predictor=RNN \
    ++task.predictor.n_layers=5 \
    ++task.predictor.h_dim=256 \
    ++trainer.enable_progress_bar=true \
    ++task.inner_epochs=50 \
    ++task.lr=0.001 \
    ++trainer.max_epochs=2000000 \
    ++callbacks.monitor=train_loss \
    ++callbacks.min_delta=-1e-4 \
    ++callbacks.patience=50 \
    ++trainer.gradient_clip_val=0.05


    python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs \
    seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14  \
    dataset=symbolic/hmm \
    task=meta_optimizer_implicit_symbolic_sequence.yaml \
    ++task.model.x_dim=50 \
    ++trainer.max_epochs=80 \
    ++logger.tags=[experiments/sgd_vs_prequential/symbolic_rnn]
