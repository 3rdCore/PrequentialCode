python train.py --multirun hydra/launcher=mila_mahan save_dir=/home/mila/m/mahan.fathi/scratch/prequential_icl/logs seed=0 dataset=regression/linear task=meta_optimizer ++logger.tags=[experiments/prequential_vs_train/regression] ++task.meta_objective=prequential,train ++task.meta_optimizer.context_aggregator._target_=models.context_aggregator.Mambaoptimizer
