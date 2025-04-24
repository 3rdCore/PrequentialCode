python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/tchebytchev \
    task=meta_optimizer \
    ++task.meta_objective=prequential,train \
    ++dataset.train_dataset.x_dim=1 \
    ++dataset.train_dataset.noise=0.0 \
    ++logger.tags=[experiments/first_experiment_on_tchebytchev]


python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs \
    seed=2\
    dataset=regression/tchebytchev \
    task=meta_optimizer_regression_tchebitchev \
    ++task.meta_objective=prequential,train \
    ++dataset.train_dataset.noise=0.1,0.2,0.3 \
    ++logger.tags=[experiments/tchebytchev] \
    ++trainer.max_epochs=400 \
    ++dataset.train_dataset.n_samples=50 \
    ++dataset.train_dataset.degree=30 \
    ++dataset.train_dataset.effective_degree=3 \
    ++dataset.val_dataset.effective_degree=3 \
    ++task.probe_n_context_points=[4,15]
