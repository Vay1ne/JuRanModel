import utils
from world import cprint
import torch
import time
import Procedure
import world
import dataloader
from model import IMP_GCN

if __name__ == '__main__':
    utils.set_seed(world.seed)
    print(">>SEED:", format(world.seed))

    if world.dataset in ['takatak', 'wechat']:
        dataset = dataloader.Loader(path="../data/" + world.dataset)
    print('===========config================')
    print(world.config)
    print("cores for test:", world.CORES)
    print("LOAD:", world.LOAD)
    print("Weight path:", world.PATH)
    print("Test Topks:", world.topks)
    print("using bpr loss")
    print('===========end===================')

    model = IMP_GCN()
