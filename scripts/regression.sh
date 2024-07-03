python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear task=meta_optimizer ++logger.tags=[debug] ++task.meta_objective=prequential,train ++task.predictor.n_layers=2 ++datamodule.batch_size=16


python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3 dataset=regression/linear task=sgd_optimizer,meta_optimizer ++logger.tags=[debug] ++task.predictor.h_dim=256 ++task.predictor.n_layers=2 ++datamodule.batch_size=16
