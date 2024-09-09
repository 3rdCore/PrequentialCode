#Meta-learner explicit
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=meta_optimizer ++tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLPConcatPredictor ++task.meta_objective=prequential,train ++task.predictor.n_layers=2,5 ++task.predictor.h_dim=512 ++datamodule.batch_size=16 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.1

#Meta-learner implicit
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=meta_optimizer_implicit ++tags=[tom/sgd_vs_preq_vs_train] ++datamodule.batch_size=16 ++dataset.train_dataset.n_samples=2000 ++dataset.train_dataset.noise=0.0,0.1

#base-SGD
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=2,5 ++task.predictor.h_dim=512 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000  ++dataset.train_dataset.noise=0.0,0.1 ++task.inner_epochs=50 callbacks=null

#Drop-out
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=2,5 ++task.predictor.h_dim=512 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000  ++dataset.train_dataset.noise=0.0,0.1 ++task.inner_epochs=50 ++task.predictor.dropout_rate=0.1,0.2,0.3,0.4  callbacks=null

#L-reg
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=2,5 ++task.predictor.h_dim=512 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000  ++dataset.train_dataset.noise=0.0,0.1 ++task.regularization_type=L2,L1 ++task.lambda_reg=0.01,0.001 ++task.inner_epochs=50 callbacks=null


#Early-Stop
python train.py --multirun hydra/launcher=mila_tom save_dir=/home/mila/t/tom.marty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=regression/linear,regression/sinusoid,regression/tchebytchev task=sgd_optimizer ++tags=[tom/sgd_vs_preq_vs_train] task/predictor=MLP ++task.predictor.n_layers=2,5 ++task.predictor.h_dim=512 ++datamodule.batch_size=64 ++dataset.train_dataset.n_samples=2000  ++dataset.train_dataset.noise=0.0,0.1
