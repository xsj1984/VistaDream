from glob import glob
from pipe.cfgs import load_cfg
from pipe.sparse_recons import Pipeline

base = f'data/bedroom'
images = glob(f'{base}/*.png')

cfg = load_cfg(f'pipe/cfgs/basic_sparse.yaml')
vistadream = Pipeline(cfg)
vistadream(images)
