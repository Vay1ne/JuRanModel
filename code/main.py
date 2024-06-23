import utils
from world import cprint
import torch
import time
from Procedure import Procedure, test, log
import world
import dataloader
from model import IMP_GCN
from logger import Log

if __name__ == '__main__':
    utils.set_seed(world.seed)
    print(">>SEED:", format(world.seed))

    dataset = dataloader.Loader()

    print('===========config================')
    print(world.config)
    print("cores for test:", world.CORES)
    print("LOAD:", world.LOAD)
    print("Weight path:", world.PATH)
    print("Test Topks:", world.topks)
    print("using bpr loss")
    print('===========end===================')

    model = IMP_GCN(dataset, world.config)
    model = model.to(world.device)
    procedure = Procedure(model, world.config)

    best_hr, best_ndcg = 0, 0
    best_epoch = 0
    count, epoch = 0, 0

    while count < 10:
        start = time.time()
        output_information = procedure.train(epoch, dataset, model)
        cprint("[valid]")
        res = test(dataset, model, 'valid', world.config['multicore'])
        hr1, ndcg1 = res['recall'][0], res['ndcg'][0]
        hr2, ndcg2 = res['recall'][0], res['ndcg'][0]
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        log.add(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        if hr1 > best_hr:
            best_epoch = epoch
            count = 0
            best_hr, best_ndcg = hr1, ndcg1
        else:
            # 小于10次退出训练
            count += 1
        epoch += 1
    log.add("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f} in invalid data".format(
        best_epoch, best_hr, best_ndcg))
    print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f} in invalid data".format(
        best_epoch, best_hr, best_ndcg))
