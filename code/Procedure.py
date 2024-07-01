import world
import numpy as np
import torch
import utils
import model
import multiprocessing
from model import IMP_GCN
from torch import nn, optim
from dataloader import Loader
from tqdm import tqdm

CORES = multiprocessing.cpu_count() // 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Procedure:
    def __init__(self,
                 model,
                 config: dict):
        self.model = model
        self.l2_w = config['l2_w']
        self.lr = config['lr']
        self.opt = optim.Adam(model.parameters(), lr=self.lr)
        self.cl_w = config['cl_w']
        self.model.log.add(config)

    def train(self, epoch, dataset, model):
        Recmodel = model
        Recmodel.train()

        S = utils.UniformSample(dataset)
        users = torch.Tensor(S[:, 0]).long()
        posVideos = torch.Tensor(S[:, 1]).long()
        negVideos = torch.Tensor(S[:, 2]).long()
        uploaders = torch.Tensor(S[:, 3]).long()

        users, posVideos, negVideos, uploaders = utils.shuffle(users, posVideos, negVideos, uploaders)

        total_batch = len(users) // world.config['bpr_batch'] + 1
        aver_loss = 0.

        train_loader = utils.minibatch(users, posVideos, negVideos, uploaders,
                                       batch_size=world.config['bpr_batch'])

        for (batch_i, data) in tqdm(enumerate(train_loader), total=total_batch):
            batch_users, batch_posVideo, batch_negVideo, batch_uploader = data[0], data[1], \
                data[2], data[3]
            # batch_users2, batch_posVlogger2, batch_negVlogger2 = data[1][0], data[1][1], data[1][2]

            batch_users = batch_users.to(world.device)
            batch_posVideo = batch_posVideo.to(world.device)
            batch_negVideo = batch_negVideo.to(world.device)
            batch_uploader = batch_uploader.to(world.device)
            bpr_loss, reg_loss, cl_loss = self.model.bpr_loss(batch_users, batch_posVideo, batch_negVideo,
                                                              batch_uploader)
            all_loss = bpr_loss + self.model.l2_w * reg_loss + self.model.cl_w * cl_loss
            if batch_i % 100 == 0:
                self.model.log.add(
                    "epoch:{}\tbatch:{}\tbpr_loss:{}\tl2_loss:{}\tcl_loss:{}\tall_loss:{}".format(epoch + 1, batch_i,
                                                                                                  bpr_loss.item(),
                                                                                                  reg_loss.item(),
                                                                                                  cl_loss.item(),
                                                                                                  all_loss.item()))
                print("epoch:{}\tbatch:{}\tbpr_loss:{}\tl2_loss:{}\tcl_loss:{}\tall_loss:{}".format(epoch + 1, batch_i,
                                                                                                    bpr_loss.item(),
                                                                                                    reg_loss.item(),
                                                                                                    cl_loss.item(),
                                                                                                    all_loss.item()))
            self.opt.zero_grad()
            all_loss.backward()
            self.opt.step()
            aver_loss += all_loss.cpu().item()

        aver_loss = aver_loss / total_batch
        # time_info = timer.dict()
        # timer.zero()
        # return f"loss{aver_loss:.3f}-{time_info}"
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {f"loss{aver_loss:.3f}"}')
        self.model.log.add(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {f"loss{aver_loss:.3f}"}')
        return f"loss{aver_loss:.3f}"

    def test(self, epoch, dataset, Recmodel, str, multicore=0):
        u_batch_size = world.config['test_batch']
        if str == 'valid':
            testDict: dict = dataset.validDict
        else:
            testDict: dict = dataset.testDict
        # eval mode with no dropout
        Recmodel = Recmodel.eval()
        max_K = max(world.topks)
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosVideos(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                # rating = rating.cpu()

                exclude_index = []
                exclude_videos = []
                for range_i, videos in enumerate(allPos):
                    exclude_index.extend([range_i] * len(videos))
                    exclude_videos.extend(videos)
                rating[exclude_index, exclude_videos] = -(1 << 10)

                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [
                #         utils.AUC(rating[i],
                #                   dataset,
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            scale = float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)

            if multicore == 1:
                pool.close()
            print(results)
            self.model.log.add(results)
            return results


def test_one_batch(x):
    sorted_videos = x[0].numpy()
    groundTrue = x[1]
    r = utils.getLabel(groundTrue, sorted_videos)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


if __name__ == '__main__':
    best_hr, best_ndcg = 0, 0
    best_epoch = 0
    count, epoch = 0, 0

    dataset = Loader()
    model = IMP_GCN(dataset, world.config)
    model = model.to(device)
    procedure = Procedure(model, world.config)
    ALL_EPOCH = 100
    print(world.config)
    # output_information = procedure.train(dataset, model)
    # print(f'EPOCH[{epoch + 1}/{ALL_EPOCH}] {output_information}')
    # procedure.log.add(f'EPOCH[{epoch + 1}/{ALL_EPOCH}] {output_information}')
    print("[valid]")
    # procedure.log.add("[valid]")
    res = procedure.test(dataset, model, 'valid', world.config['multicore'])
    hr1, ndcg1 = res['recall'][0], res['ndcg'][0]
    hr2, ndcg2 = res['recall'][0], res['ndcg'][0]
    # if hr1 > best_hr:
    #     best_epoch = epoch
    #     count = 0
    #     best_hr, best_ndcg = hr1, ndcg1
    # epoch += 1
# print("[test]")
# res = test(dataset, model, 'test', world.config['multicore'])
