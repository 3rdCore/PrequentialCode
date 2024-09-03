#Meta-learner
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=meta_optimizer ++logger.tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLPConcatPredictor ++task.meta_objective=prequential,train ++task.predictor.n_layers=5 ++task.predictor.h_dim=1024 ++datamodule.batch_size=16 ++task.dataset.train_dataset.n_samples=2000

#base-SGD
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++logger.tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=1024 ++datamodule.batch_size=1 ++task.dataset.train_dataset.n_samples=2000

#L-reg
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++logger.tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=1024 ++task.regularization_type=L2,L1 ++task.lambda_reg=0.1,0.01,0.001 ++datamodule.batch_size=1 ++task.dataset.train_dataset.n_samples=2000

#Drop-out
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++logger.tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=1024 ++task.predictor.dropout_rate=0.1,0.2,0.3 ++datamodule.batch_size=1 ++task.dataset.train_dataset.n_samples=2000
