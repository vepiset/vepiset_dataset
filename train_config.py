
from easydict import EasyDict as edict
from lib.utils.seed_utils import seed_everything

config = edict()

config.TRAIN = edict()

config.TRAIN.process_num = 4

config.TRAIN.batch_size = 128
config.TRAIN.validatiojn_batch_size = config.TRAIN.batch_size
config.TRAIN.accumulation_batch_size = 128
config.TRAIN.log_interval = 10
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 20

config.TRAIN.init_lr = 0.0005
config.TRAIN.lr_scheduler = 'cos'

if config.TRAIN.lr_scheduler == 'ReduceLROnPlateau':
    config.TRAIN.epoch = 100
    config.TRAIN.lr_scheduler_factor = 0.1

config.TRAIN.weight_decay_factor = 1.e-2
config.TRAIN.vis = False


config.TRAIN.warmup_step = 1500
config.TRAIN.opt = 'Adamw'

config.TRAIN.gradient_clip = 5

config.TRAIN.vis_mixcut = False
if config.TRAIN.vis:
    config.TRAIN.mix_precision = False
else:
    config.TRAIN.mix_precision = False

config.MODEL = edict()


config.MODEL.model_path = './models/_base_line/'

config.DATA = edict()

config.DATA.data_file = '/data/tt.csv'

config.DATA.data_root_path = 'utils'

config.MODEL.early_stop = 20

config.MODEL.pretrained_model = None

config.SEED = 10086

seed_everything(config.SEED)
