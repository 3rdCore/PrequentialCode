python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer \
    ++task.meta_objective=prequential \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/single_tokenization] &


python train.py --multirun hydra/launcher=mila_tejas save_dir=/home/mila/t/tejas.kasetty/scratch/prequential_icl/logs \
    seed=0,1,2,3,4 \
    dataset=regression/sinusoid \
    task=meta_optimizer_implicit \
    ++task.model._target_=models.implicit.DecoderTransformer2 \
    ++logger.tags=[experiments/icl_architectures/regression,experiments/single_tokenization]
