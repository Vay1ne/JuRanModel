import utils
from world import cprint
import torch
import time
from Procedure import Procedure
import world
import dataloader
from model import IMP_GCN

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
    weight_file = utils.getFileName()

    best_hr, best_ndcg = 0, 0
    best_epoch = 0
    count, epoch = 0, 0
    model_dir = weight_file + str(epoch) + '.model'
    while count < 10:
        start = time.time()
        output_information = procedure.train(epoch, dataset, model)
        cprint("[valid]")
        res = procedure.test(epoch, dataset, model, 'valid', world.config['multicore'])
        hr1, ndcg1 = res['recall'][0], res['ndcg'][0]
        hr2, ndcg2 = res['recall'][0], res['ndcg'][0]

        # print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        if hr1 > best_hr:
            best_epoch = epoch
            count = 0
            best_hr, best_ndcg = hr1, ndcg1
            model_dir = weight_file + str(epoch) + '.model'
            torch.save(model.state_dict(), model_dir)
        else:
            # 小于10次退出训练
            count += 1
        epoch += 1
    print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f} in invalid data".format(
        best_epoch + 1, best_hr, best_ndcg))
    print("save to" + model_dir)

    # test
    model.load_state_dict(torch.load(weight_file + str(best_epoch) + '.model'))
    cprint("[test]")
    res = procedure.test(epoch, model, 'test', world.config['multicore'])
