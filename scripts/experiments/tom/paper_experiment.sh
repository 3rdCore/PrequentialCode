
#########################################################
#3D linear
#########################################################

#base-SGD
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/linear ++dataset.train_dataset.x_dim=3 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 &

#L1 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/linear ++dataset.train_dataset.x_dim=3 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L1 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

# L2 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/linear ++dataset.train_dataset.x_dim=3 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L2 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

# Early-Stop
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/linear ++dataset.train_dataset.x_dim=3 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=val_loss ++callbacks.min_delta=1e-3 ++callbacks.patience=10 ++trainer.gradient_clip_val=0.05 &

#Dropout
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/linear ++dataset.train_dataset.x_dim=3 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task=predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.dropout=0.2,0.4 &


#########################################################
#1D sinusoid
#########################################################

#base-SGD
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/sinusoid ++dataset.train_dataset.x_dim=1 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 &

#L1 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/sinusoid ++dataset.train_dataset.x_dim=1 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L1 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

#L2 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/sinusoid ++dataset.train_dataset.x_dim=1 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L2 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

# Early-Stop
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/sinusoid ++dataset.train_dataset.x_dim=1 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task/predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=val_loss ++callbacks.min_delta=1e-3 ++callbacks.patience=10 ++trainer.gradient_clip_val=0.05 &

#Dropout
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=regression/sinusoid ++dataset.train_dataset.x_dim=1 task=sgd_optimizer ++tags=[tom/sgd_benchmark,tom/final_benchy] task=predictor=MLP ++task.predictor.n_layers=5 ++task.predictor.h_dim=64 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.dropout=0.2,0.4 &

#########################################################
#mastermind
#########################################################

#base-SGD
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=symbolic/mastermind task=sgd_optimizer ++task.loss_fn._target_=utils.CrossEntropyLossFlat ++predictor.in_features=48 ++predictor.out_features=18 ++predictor.n_layers=5 ++predictor.h_dim=256 ++dataset.train_dataset.one_hot_y=False ++tags=[tom/sgd_benchmark,tom/final_benchy] ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=3000 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 &

#L1 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=symbolic/mastermind task=sgd_optimizer ++task.loss_fn._target_=utils.CrossEntropyLossFlat ++predictor.in_features=48 ++predictor.out_features=18 ++predictor.n_layers=5 ++predictor.h_dim=256 ++dataset.train_dataset.one_hot_y=False ++tags=[tom/sgd_benchmark,tom/final_benchy] ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=3000 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L1 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

#L2 reg.
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=symbolic/mastermind task=sgd_optimizer ++task.loss_fn._target_=utils.CrossEntropyLossFlat ++predictor.in_features=48 ++predictor.out_features=18 ++predictor.n_layers=5 ++predictor.h_dim=256 ++dataset.train_dataset.one_hot_y=False ++tags=[tom/sgd_benchmark,tom/final_benchy] ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=3000 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.regularization_type=L2 ++task.lambda_reg=0.1,0.05,0.01,0.005 &

# Early-Stop
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=symbolic/mastermind task=sgd_optimizer ++task.loss_fn._target_=utils.CrossEntropyLossFlat ++predictor.in_features=48 ++predictor.out_features=18 ++predictor.n_layers=5 ++predictor.h_dim=256 ++dataset.train_dataset.one_hot_y=False ++tags=[tom/sgd_benchmark,tom/final_benchy] ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=3000 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=val_loss ++callbacks.min_delta=1e-3 ++callbacks.patience=10 ++trainer.gradient_clip_val=0.05 &

#Dropout
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 dataset=symbolic/mastermind task=sgd_optimizer ++task.loss_fn._target_=utils.CrossEntropyLossFlat ++predictor.in_features=48 ++predictor.out_features=18 ++predictor.n_layers=5 ++predictor.h_dim=256 ++dataset.train_dataset.one_hot_y=False ++tags=[tom/sgd_benchmark,tom/final_benchy] ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=3000 ++task.inner_epochs=2000 ++task.lr=0.0001 ++trainer.max_epochs=2000000 ++callbacks.monitor=train_loss ++callbacks.min_delta=-1e-4 ++callbacks.patience=50 ++trainer.gradient_clip_val=0.05 ++task.dropout=0.2,0.4 &


#########################################################
#Old Stuff
#Meta-learner explicit
#python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid x_dim=1,3 task=meta_optimizer ++tags=[tom/sgd_benchmark] task/predictor=MLPConcatPredictor ++task.meta_objective=prequential,train ++task.predictor.n_layers=5 ++task.predictor.h_dim=512 ++datamodule.batch_size=16 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2

#Meta-learner implicit
#python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid x_dim=1,   3 task=meta_optimizer_implicit ++tags=[tom/sgd_benchmark] ++datamodule.batch_size=16 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.2
