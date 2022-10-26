import os
from datetime import datetime
from easydict import EasyDict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cfg = EasyDict()

cfg.lr = 1e-5
cfg.embed_size = 128
cfg.alpha = 0.20

cfg.max_epoch = 3
cfg.device = 'gpu'

cfg.eval_batch_size = 1
cfg.eval_top_k = (1, 3, 5, 10)

cfg.evaluate_before_training = False
cfg.epoch_to_load = None

# dataset
cfg.data_dir = r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\merged_data'

# checkpoint
cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'data/checkpoints/image_retrieval')

# Dataloader params
cfg.m, cfg.l = 16, 4

# logger params
cfg.experiment_name = f"embed_{cfg.embed_size}_lr-{cfg.lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
cfg.logdir = os.path.join(ROOT_DIR, 'data/logs/resnet50')
cfg.embeddings_path = os.path.join(cfg.logdir, cfg.experiment_name, 'embeddings')



