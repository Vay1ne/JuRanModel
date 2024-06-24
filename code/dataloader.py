import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def n_videos(self):
        raise NotImplementedError

    @property
    def n_uploaders(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def trainDataSize2(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def uploader_list(self):
        raise NotImplementedError

    @property
    def allPos2(self):
        raise NotImplementedError

    def getUserVideoFeedback(self, users, videos):
        raise NotImplementedError

    def getUserPosVideos(self, users):
        raise NotImplementedError

    def getUserNegVideos(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg videos in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config):
        # train or test
        path = '../data/' + config['dataset']
        cprint(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1, "valid": 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.n_video = 0
        self.n_uploader = 0
        user_video_file = path + '/user_video_data.txt'
        user_uploader_file = path + '/user_uploader_data.txt'
        uploader_video_file = path + '/uploader_video_data.txt'
        size_file = path + '/data_size.txt'
        self.path = path
        trainUniqueUsers, trainVideo, trainUser, trainUploader = [], [], [], []
        testUniqueUsers, testVideo, testUser = [], [], []
        validUniqueUsers, validVideo, validUser = [], [], []

        self.traindataSize = 0
        self.validDataSize = 0
        self.testDataSize = 0

        with open(size_file) as f:
            self.n_user, self.n_video, self.n_uploader = [int(s) for s in f.readline().split('\t')][:3]

        # u-i train
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[1:-2]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(videos))
                    trainVideo.extend(videos)
                    self.traindataSize += len(videos)
        self.trainUniqueUsers = np.array(trainUniqueUsers)  # 用户数
        self.trainUser = np.array(trainUser)  # 大小==训练集中user-video交互数
        self.trainVideo = np.array(trainVideo)  # 同上

        # u-i valid
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[-2:-1]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(videos))
                    validVideo.extend(videos)
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validVideo = np.array(validVideo)

        # u-i test
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[-1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(videos))
                    testVideo.extend(videos)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testVideo = np.array(testVideo)

        # u-a
        with open(user_uploader_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uploaders = [int(i) for i in l[1:-2]]
                    uid = int(l[0])
                    # trainUniqueUsers.append(uid)  # 注销这里，否则trainUniqueUsers会多一倍数据
                    trainUser.extend([uid] * len(uploaders))
                    trainUploader.extend(uploaders)
        self.trainUploader = np.array(trainUploader)

        # 读取包含uploader和视频配对关系的文件
        with open(uploader_video_file, 'r') as f:
            # 将文件中的每一行转换为一个包含uploader和视频ID的元组，并存入列表
            self.A_I_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(' ')), f.readlines()))
        # 将元组列表转换为numpy整数数组
        indice = np.array(self.A_I_pairs, dtype=np.int32)
        # 创建一个与uploader视频对数目相同的全为1的浮点数组
        values = np.ones(len(self.A_I_pairs), dtype=np.float32)
        # 使用indices和values数组创建一个COO格式的稀疏矩阵，然后转换为CSR格式。
        # 这个矩阵表示uploader和视频之间的真实关联关系
        self.ground_truth_a_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.n_uploader, self.n_video)).tocsr()

        # 按视频ID对uploader视频对进行排序
        self.A_I_pairs = sorted(self.A_I_pairs, key=lambda x: x[-1])

        # 创建一个字典，键为视频ID，值为uploader ID
        self.i_a_dict = dict(list(map(lambda x: x[::-1], self.A_I_pairs)))
        # 从字典中提取uploader ID的列表
        self._uploaderList = list(self.i_a_dict.values())

        self.uploader_dict = {}
        with open(uploader_video_file, 'r') as file:
            for line in file:
                # 去掉行末的换行符并拆分成uploader_id和video_id
                uploader_id, video_id = line.strip().split(' ')

                # 将uploader_id和video_id转换为整数类型
                uploader_id = int(uploader_id)
                video_id = int(video_id)

                # 如果uploader_id不在字典中，则初始化一个空列表
                if uploader_id not in self.uploader_dict:
                    self.uploader_dict[uploader_id] = []

                # 将video_id添加到对应的uploader_id的列表中
                self.uploader_dict[uploader_id].append(video_id)

        # 创建用户和视频的二分图
        self.UserVideoNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainVideo)),
                                       shape=(self.n_user, self.n_video))
        # 计算每个用户关联的视频数量，若数量为0则设为1
        self.users_D = np.array(self.UserVideoNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        # 计算每个视频关联的用户数量，若数量为0则设为1
        self.videos_D = np.array(self.UserVideoNet.sum(axis=0)).squeeze()
        self.videos_D[self.videos_D == 0.] = 1.

        # 创建用户和uploader的二分图
        self.UseruploaderNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainUploader)),
                                          shape=(self.n_user, self.n_uploader))
        # 计算每个用户关联的uploader数量，若数量为0则设为1
        self.users2_D = np.array(self.UseruploaderNet.sum(axis=1)).squeeze()
        self.users2_D[self.users2_D == 0.] = 1
        # 计算每个uploader关联的用户数量，若数量为0则设为1
        self.uploaders_D = np.array(self.UseruploaderNet.sum(axis=0)).squeeze()
        self.uploaders_D[self.uploaders_D == 0.] = 1.

        # 预先计算每个用户关联的所有视频
        self._allPos = self.getUserPosVideos(list(range(self.n_user)))
        # 预先计算每个用户关联的所有uploader
        self._allPos2 = self.getUserPosuploaders(list(range(self.n_user)))
        # 构建验证集字典
        self.__validDict = self.__build_valid()
        # 构建测试集字典
        self.__testDict = self.__build_test()
        # 输出数据集准备就绪的信息
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def n_videos(self):
        return self.n_video

    @property
    def n_uploaders(self):
        return self.n_uploader

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def uploader_list(self):
        return self._uploaderList

    @property
    def allPos2(self):
        return self._allPos2

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        try:
            pre_adj_mat = sp.load_npz(self.path + '/user_video_s_pre_adj_mat.npz')
            print("successfully loaded...")
            user_video_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            user_video_graph = user_video_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/user_uploader_s_pre_adj_mat.npz')
            print("successfully loaded...")
            user_uploader_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            user_uploader_graph = user_uploader_graph.coalesce().to(world.device)

            # pre_adj_mat = sp.load_npz(self.path + '/pooling1_s_pre_adj_mat.npz')
            # print("successfully loaded...")
            # pool1_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            # pool1_graph = pool1_graph.coalesce().to(world.device)
            #
            # pre_adj_mat = sp.load_npz(self.path + '/pooling2_s_pre_adj_mat.npz')
            # print("successfully loaded...")
            # pool2_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            # pool2_graph = pool2_graph.coalesce().to(world.device)
            #
            # pre_adj_mat = sp.load_npz(self.path + '/pooling3_s_pre_adj_mat.npz')
            # print("successfully loaded...")
            # pool3_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            # pool3_graph = pool3_graph.coalesce().to(world.device)

        except:
            print("generating adjacency matrix")

            # user_video_adj, user_uploader_adj = self.generate_user_video_uploader_user_adj(self.UserVideoNet,
            #                                                                                self.UseruploaderNet,
            #                                                                                self.ground_truth_a_i, p=0.4,
            #                                                                                q=0.1
            #                                                                                )
            user_video_adj, user_uploader_adj = self.UserVideoNet, self.UseruploaderNet
            # u-i graph
            adj_mat = sp.dok_matrix((self.n_user + self.n_video, self.n_user + self.n_video), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = user_video_adj.tolil()

            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            sp.save_npz(self.path + '/user_video_s_pre_adj_mat.npz', norm_adj)
            user_video_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            user_video_graph = user_video_graph.coalesce().to(world.device)

            # u-a graph
            adj_mat = sp.dok_matrix((self.n_user + self.n_uploader, self.n_user + self.n_uploader), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = user_uploader_adj.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/user_uploader_s_pre_adj_mat.npz', norm_adj)
            user_uploader_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            user_uploader_graph = user_uploader_graph.coalesce().to(world.device)

            # # pooling garph1 (a-i)
            # bundle_size = self.ground_truth_a_i.sum(axis=1) + 1e-8
            # ai_graph = sp.diags(1 / bundle_size.A.ravel()) @ self.ground_truth_a_i
            # sp.save_npz(self.path + '/pooling1_s_pre_adj_mat.npz', ai_graph)
            # pool1_graph = self._convert_sp_mat_to_sp_tensor(ai_graph)
            # pool1_graph = pool1_graph.coalesce().to(world.device)
            #
            # #  pooling graph2 (i-a)
            # sp.save_npz(self.path + '/pooling2_s_pre_adj_mat.npz', ai_graph.T)
            # pool2_graph = self._convert_sp_mat_to_sp_tensor(ai_graph.T)
            # pool2_graph = pool2_graph.coalesce().to(world.device)
            #
            # # pooling graph3 (i-u)
            # video_size = self.UserVideoNet.sum(axis=1) + 1e-8
            # u_i_graph = sp.diags(1 / video_size.A.ravel()) @ self.UserVideoNet
            # sp.save_npz(self.path + '/pooling3_s_pre_adj_mat.npz', u_i_graph.T)
            # pool3_graph = self._convert_sp_mat_to_sp_tensor(u_i_graph.T)
            # pool3_graph = pool3_graph.coalesce().to(world.device)

        # self.Graph = [user_video_graph, user_uploader_graph, pool1_graph, pool2_graph, pool3_graph]
        self.Graph = [user_video_graph, user_uploader_graph]

        return self.Graph

    def generate_user_video_uploader_user_adj(self, user_video_adj, user_uploader_adj, uploader_video_adj, p, q):
        user_video_adj = user_video_adj.tolil()
        user_uploader_adj = user_uploader_adj.tolil()
        uploader_video_adj = uploader_video_adj.tolil()

        new_user_video_adj = user_video_adj.copy()
        new_user_uploader_adj = user_uploader_adj.copy()
        for user in range(self.n_user):
            if np.random.random() <= p:
                current_node = np.random.choice(user_uploader_adj[user].nonzero()[1])
                # U-A-U
                if np.random.random() <= q:
                    # get users interacted with uploader
                    neighbors = np.array(user_uploader_adj.T[current_node].nonzero()[1])
                    neighbors = neighbors[~np.in1d(neighbors, user)]
                    current_node = np.random.choice(neighbors)
                    related_videos = user_video_adj[user].nonzero()[1]
                    for video in related_videos:
                        new_user_video_adj[current_node, video] += 1
                # U-A-I
                else:
                    # get videos interacted with uploader
                    neighbors = np.array(uploader_video_adj[current_node].nonzero()[1])
                    current_node = np.random.choice(neighbors)
                    new_user_video_adj[user, current_node] += 1

            if np.random.random() <= p:
                current_node = np.random.choice(user_video_adj[user].nonzero()[1])  # 随机选择一个已交互的video作为起点
                # U-I-U
                if np.random.random() <= q:
                    # get users interacted with videos
                    neighbors = np.array(user_video_adj.T[current_node].nonzero()[1])
                    neighbors = neighbors[~np.in1d(neighbors, user)]
                    current_node = np.random.choice(neighbors)
                    related_uploaders = user_uploader_adj[user].nonzero()[1]
                    for uploader in related_uploaders:
                        new_user_uploader_adj[current_node, uploader] += 1

        return new_user_video_adj.tocsr(), new_user_uploader_adj.tocsr()

    def __build_valid(self):
        """
        return:
            dict: {user: [videos]}
        """
        valid_data = {}
        for i, video in enumerate(self.validVideo):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(video)
            else:
                valid_data[user] = [video]
        return valid_data

    def __build_test(self):
        """
        return:
            dict: {user: [videos]}
        """
        test_data = {}
        for i, video in enumerate(self.testVideo):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(video)
            else:
                test_data[user] = [video]
        return test_data

    def getUserVideoFeedback(self, users, videos):
        """
        users:
            shape [-1]
        videos:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserVideoNet[users, videos])
        return np.array(self.UserVideoNet[users, videos]).astype('uint8').reshape((-1,))

    def getUserPosVideos(self, users):
        posVideos = []
        for user in users:
            posVideos.append(self.UserVideoNet[user].nonzero()[1])
        return posVideos

    def getUserPosuploaders(self, users):
        posuploaders = []
        for user in users:
            posuploaders.append(self.UseruploaderNet[user].nonzero()[1])
        return posuploaders


if __name__ == '__main__':
    loader = Loader(world.config)
    loader.getSparseGraph()
