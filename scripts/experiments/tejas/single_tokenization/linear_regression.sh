python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/linear \
    task=meta_optimizer \
    ++task.meta_objective=prequential \
    ++dataset.train_dataset.x_dim=3 \
    ++dataset.train_dataset.noise=0.2 \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/single_tokenization] &

python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/linear \
    task=meta_optimizer_implicit \
    ++dataset.train_dataset.x_dim=3 \
    ++dataset.train_dataset.noise=0.2 \
    ++task.model._target_=models.implicit.DecoderTransformer2 \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/single_tokenization]
