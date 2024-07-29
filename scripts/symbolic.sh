#python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs seed=0,1,2,3,4 dataset=symbolic/arc task=meta_optimizer  ++logger.tags=[debug] task/predictor=MLPConcatPredictor ++task.meta_objective=prequential,train ++task.predictor.h_dim=512 ++task.predictor.n_layers=2 ++datamodule.batch_size=16
# simple test run
python train.py hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs seed=0 dataset=symbolic/arc task=meta_optimizer_symbolic "++logger.tags=[debug,symbolic]" ++task.meta_objective=prequential

python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs seed=0 dataset=symbolic/arc task=meta_optimizer_symbolic  ++logger.tags=[SYMBOLIC] task/predictor=MLP ++task.meta_objective=prequential,train ++task.predictor.h_dim=256 ++task.predictor.n_layers=2 ++datamodule.batch_size=128
