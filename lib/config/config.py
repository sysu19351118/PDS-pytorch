from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'
cfg.ff_num = 5
cfg.layer1rate=[0.0,0.1]
cfg.ifmultistage=True
cfg.isrec=True
cfg.ct_weight = 0.1
cfg.wh_weight = 0.2
cfg.replace_box = True

# network
cfg.network = 'dla_34'
cfg.evolution_num =30

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8
cfg.data_path = '/data/tzx/data/images'

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 30
cfg.eval_ep = 5

cfg.use_gt_det = False

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''
cfg.ff_num=0


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    #cfg.model_dir = os.path.join(cfg.model_dir, cfg.workname, cfg.model)
    #cfg.record_dir = os.path.join(cfg.record_dir, cfg.workname, cfg.model)
    #cfg.result_dir = os.path.join(cfg.result_dir, cfg.workname, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="config.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('-f', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
