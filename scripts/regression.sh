python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear task=meta_optimizer ++logger.tags=[debug] task/predictor=MLPConcatPredictor ++task.meta_objective=prequential,train ++task.predictor.n_layers=2 ++datamodule.batch_size=16


python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3 dataset=regression/linear,regression/sinusoid,regression/mlp task=sgd_optimizer ++logger.tags=[SGD-META] task/predictor=MLP ++task.predictor.h_dim=64 ++task.predictor.n_layers=2 ++datamodule.batch_size=8,16,32


python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1 dataset=regression/linear,regression/sinusoid task=meta_optimizer ++logger.tags=[HypSearch/metaopt-expressivity] ++task.meta_objective=prequential,train task/predictor=MLPConcatPredictor ++task.predictor.h_dim=512 ++task.predictor.n_layers=5 ++datamodule.batch_size=16 ++task.context_aggregator.n_layers=4,8,12 ++task.context_aggregator.z_dim=16,32,128 ++task.context_aggregator.h_dim=256,512,1024 ++task.context_aggregator.n_heads=4,8
