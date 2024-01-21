from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
import torch.multiprocessing
import numpy as np
import random
import os
import sys
import time




def train(cfg, network):
    
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    
    begin_epoch = 0

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

    return network


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"



def main():
    time1 = time.time()
    init_seed(1)
    network = make_network(cfg)
    network = network.cuda()

    train(cfg, network)

    print("训练时间：",time.time()-time1)


if __name__ == "__main__":
    main()
