import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from logger import Log
from utils import cust_mul
from dataloader import Loader
from world import config
from time import strftime, localtime, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectionHead(nn.Module):
    def __init__(
            self,
            config: dict,
    ):
        super().__init__()
        self.config = config
        projection_dim = 16
        # dropout = self.config['projection_dropout']
        dropout = 0.5
        embedding_dim = self.config['latent_dim']
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class IMP_GCN(nn.Module):
    def __init__(self,
                 dataset, config):
        super(IMP_GCN, self).__init__()
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']
        self.dropout_bool = config['dropout_bool']
        self.keep_prob = config['keep_prob']
        self.Graph = dataset.getSparseGraph()
        self.num_users = dataset.n_user
        self.num_uploaders = dataset.n_uploaders
        self.num_videos = dataset.n_videos
        self.uploader_video_dict = dataset.uploader_dict
        self.video_uploader_dict = dataset.i_a_dict
        self.groups = config['groups']
        self.device = device
        self.l2_w = config['l2_w']
        self.cl_w = config['cl_w']
        self.cl_temp = config['cl_temp']
        self.single = config['single']
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log('JuRanModel', 'JuRanModel' + ' ' + current_time)

        self.__init_weight()

        self.projection = ProjectionHead(config)
        self.userProjection = ProjectionHead(config)
        self.videoProjection = ProjectionHead(config)
        self.uploaderProjection = ProjectionHead(config)

    def __init_weight(self):
        self.embedding_user_ua = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user_uv = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_uploader = torch.nn.Embedding(num_embeddings=self.num_uploaders, embedding_dim=self.latent_dim)
        self.embedding_video = torch.nn.Embedding(num_embeddings=self.num_videos, embedding_dim=self.latent_dim)

        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc_g = torch.nn.Linear(self.latent_dim, self.groups)
        self.f = nn.Sigmoid()

        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_uploader.weight, std=0.1)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_uploader.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_video.weight, gain=1)

        nn.init.normal_(self.embedding_user_ua.weight, std=0.1)
        nn.init.normal_(self.embedding_user_uv.weight, std=0.1)
        nn.init.normal_(self.embedding_uploader.weight, std=0.1)
        nn.init.normal_(self.embedding_video.weight, std=0.1)
        # nn.init.xavier_uniform_(self.fc.weight, gain=1)
        # nn.init.xavier_uniform_(self.fc_g.weight, gain=1)
        print('1')

    def __dropout_x(self, x, keep_prob):
        # 获取输入张量的大小
        size = x.size()

        # 获取稀疏张量的索引（坐标）
        index = x.indices().t()

        # 获取稀疏张量的值
        values = x.values()

        # 生成一个与values长度相同的随机张量，每个元素在[0, 1)之间
        random_index = torch.rand(len(values)) + keep_prob

        # 将随机张量的值加上keep_prob，然后将其转换为整数，再转换为布尔类型
        # 这样可以生成一个布尔张量，其中大约有keep_prob比例的元素为True
        random_index = random_index.int().bool()

        # 根据布尔索引筛选出保留下来的索引
        index = index[random_index]

        # 根据布尔索引筛选出保留下来的值，并除以keep_prob进行缩放
        values = values[random_index] / keep_prob

        # 使用保留下来的索引和值，重新构造一个稀疏张量
        g = torch.sparse.FloatTensor(index.t(), values, size)

        # 返回新的稀疏张量
        return g

    def __dropout(self, keep_prob):
        # graph = self.__dropout_x(self.Graph, keep_prob)
        # return graph
        graph = []
        length = len(self.Graph)
        for i in range(length):
            graph.append(self.__dropout_x(self.Graph[i], keep_prob))
        return graph

    def computer_ua(self, dropout_bool):
        # 获取用户和项目的嵌入权重
        users_emb = self.embedding_user_ua.weight
        uploaders_emb = self.embedding_uploader.weight
        # 将用户和项目的嵌入拼接在一起
        all_emb = torch.cat([users_emb, uploaders_emb])

        # 如果在训练模式并且使用dropout，则应用dropout到图结构
        if dropout_bool and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            # 否则使用原始的图结构
            g_droped = self.Graph

        # 计算自有嵌入和邻居嵌入
        ego_embed = all_emb
        side_embed = torch.sparse.mm(g_droped[1], all_emb)  # 一阶特征

        # 通过全连接层和激活函数计算临时嵌入
        temp = self.dropout(self.leaky(self.fc(ego_embed + side_embed)))
        # 计算分组得分
        group_scores = self.dropout(self.fc_g(temp))

        # 获取得分最高的组索引
        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)
        one_hot_emb = torch.eq(group_scores, a_top).float()

        # 分别获取用户和项目的one-hot嵌入表示
        u_one_hot, i_one_hot = torch.split(one_hot_emb, [self.num_users, self.num_uploaders])
        # 将项目的one-hot嵌入表示设置为全1
        u_one_hot = torch.ones(u_one_hot.shape).to(self.device)
        # 将用户和项目的one-hot嵌入表示拼接在一起并转置
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

        # # 获取得分最高的组索引，针对上传者进行分组
        # uploaders_top, uploaders_top_idx = torch.topk(group_scores[self.num_users:], k=1, sorted=False)
        # uploaders_one_hot_emb = torch.eq(group_scores[self.num_users:], uploaders_top).float()
        #
        # # 分别获取用户和项目的one-hot嵌入表示
        # u_one_hot = torch.ones((self.num_users, 1)).to(self.device)  # 用户的one-hot嵌入表示设置为全1
        # i_one_hot = uploaders_one_hot_emb
        # # 将用户和项目的one-hot嵌入表示拼接在一起并转置
        # one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

        # 创建子图列表
        subgraph_list = []
        for g in range(self.groups):
            # 通过元素乘法生成子图
            temp = cust_mul(g_droped[1], one_hot_emb[g], 1)
            temp = cust_mul(temp, one_hot_emb[g], 0)
            subgraph_list.append(temp)

        # 初始化所有层的嵌入列表
        all_emb_list = [[None for _ in range(self.groups)] for _ in range(self.n_layers)]
        for g in range(0, self.groups):
            # 第0层的嵌入为自有嵌入
            all_emb_list[0][g] = ego_embed

        # 计算每一层的嵌入
        for k in range(1, self.n_layers):
            for g in range(self.groups):
                # 通过子图计算每一层的嵌入
                all_emb_list[k][g] = torch.sparse.mm(subgraph_list[g], all_emb_list[k - 1][g])

        # 对每一层的嵌入进行求和
        all_emb_list = [torch.sum(torch.stack(x), 0) for x in all_emb_list]

        # 如果使用单层嵌入，直接取最后一层的嵌入
        if self.single:
            all_emb = all_emb_list[-1]
        else:
            # 否则对所有层的嵌入进行加权求和
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            all_emb_list = [x * w for x, w in zip(all_emb_list, weights)]
            all_emb = torch.sum(torch.stack(all_emb_list), 0)
            # all_emb = torch.mean(torch.stack(all_emb_list),0)
            # all_emb = all_emb_list[-1]

        # 分别获取用户和项目的嵌入
        users, uploaders = torch.split(all_emb, [self.num_users, self.num_uploaders])
        return users, uploaders

    def computer_uv(self, dropout_bool):
        # 获取用户和项目的嵌入权重
        users_emb = self.embedding_user_uv.weight
        videos_emb = self.embedding_video.weight
        # 将用户和项目的嵌入拼接在一起
        all_emb = torch.cat([users_emb, videos_emb])

        # 如果在训练模式并且使用dropout，则应用dropout到图结构
        if dropout_bool and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            # 否则使用原始的图结构
            g_droped = self.Graph

        # 计算自有嵌入和邻居嵌入
        ego_embed = all_emb
        side_embed = torch.sparse.mm(g_droped[0], all_emb)

        # 通过全连接层和激活函数计算临时嵌入
        temp = self.dropout(self.leaky(self.fc(ego_embed + side_embed)))
        # 计算分组得分
        group_scores = self.dropout(self.fc_g(temp))

        # 获取得分最高的组索引
        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)
        one_hot_emb = torch.eq(group_scores, a_top).float()

        # 分别获取用户和项目的one-hot嵌入表示
        u_one_hot, i_one_hot = torch.split(one_hot_emb, [self.num_users, self.num_videos])
        # 将项目的one-hot嵌入表示设置为全1
        i_one_hot = torch.ones(i_one_hot.shape).to(self.device)
        # 将用户和项目的one-hot嵌入表示拼接在一起并转置
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

        # 创建子图列表
        subgraph_list = []
        for g in range(self.groups):
            # 通过元素乘法生成子图
            temp = cust_mul(g_droped[0], one_hot_emb[g], 1)
            temp = cust_mul(temp, one_hot_emb[g], 0)
            subgraph_list.append(temp)

        # 初始化所有层的嵌入列表
        all_emb_list = [[None for _ in range(self.groups)] for _ in range(self.n_layers)]
        for g in range(0, self.groups):
            # 第0层的嵌入为自有嵌入
            all_emb_list[0][g] = ego_embed

        # 计算每一层的嵌入
        for k in range(1, self.n_layers):
            for g in range(self.groups):
                # 通过子图计算每一层的嵌入
                all_emb_list[k][g] = torch.sparse.mm(subgraph_list[g], all_emb_list[k - 1][g])

        # 对每一层的嵌入进行求和
        all_emb_list = [torch.sum(torch.stack(x), 0) for x in all_emb_list]

        # 如果使用单层嵌入，直接取最后一层的嵌入
        if self.single:
            all_emb = all_emb_list[-1]
        else:
            # 否则对所有层的嵌入进行加权求和
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            all_emb_list = [x * w for x, w in zip(all_emb_list, weights)]
            all_emb = torch.sum(torch.stack(all_emb_list), 0)
            # all_emb = torch.mean(torch.stack(all_emb_list),0)
            # all_emb = all_emb_list[-1]

        # 分别获取用户和项目的嵌入
        users, videos = torch.split(all_emb, [self.num_users, self.num_videos])
        return users, videos

    def compute_uploader_embedding(self, video_embedding):
        uploader_emb = torch.Tensor(self.num_uploaders, self.latent_dim)
        # 统计每个uploader的视频数量，用于后续求平均
        uploader_count = {}

        # 遍历uploader_dict，聚合每个uploader发布的视频的embedding
        for uploader_id, video_ids in self.uploader_video_dict.items():
            # 初始化一个累加变量
            sum_emb = torch.zeros_like(video_embedding[0])
            for video_id in video_ids:
                sum_emb += video_embedding[video_id]

            # 计算平均值
            uploader_emb[uploader_id] = sum_emb / len(video_ids)
            uploader_count[uploader_id] = len(video_ids)
        uploader_emb = uploader_emb.to(device)
        return uploader_emb

    def compute(self):
        user_0, uploader_0 = self.computer_ua(self.dropout_bool)
        user_1, uploader_1 = self.computer_ua(1)
        user_2, uploader_2 = self.computer_ua(1)
        user_3, video_0 = self.computer_uv(self.dropout_bool)
        uploader_3 = self.compute_uploader_embedding(video_0)

        return [user_0, user_1, user_2, user_3], [uploader_0, uploader_1, uploader_2, uploader_3], [video_0]

    def getUsersRating(self, users):
        all_users, all_uploaders, all_videos = self.compute()
        users_emb = (all_users[0][users] + all_users[3][users]) / 2
        items_emb = all_videos[0]
        uploaders_emb = (all_uploaders[0] + all_uploaders[3]) / 2
        videos_uploader_emb = torch.zeros_like(items_emb)
        for uploader_id, video_ids in self.uploader_video_dict.items():
            for video_id in video_ids:
                videos_uploader_emb[video_id] = uploaders_emb[uploader_id]
        items_emb = (videos_uploader_emb + items_emb) / 2
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_videos, neg_videos, uploaders, pos_uploaders, neg_uploaders):
        # 调用computer方法，计算所有用户和项目的嵌入
        all_users, all_uploaders, all_videos = self.compute()

        users_emb0 = all_users[0][users]
        users_emb1 = all_users[1][users]
        users_emb2 = all_users[2][users]
        users_emb3 = all_users[3][users]

        pos_emb = all_videos[0][pos_videos]
        neg_emb = all_videos[0][neg_videos]

        uploaders_emb0 = all_uploaders[0][uploaders]
        uploaders_emb1 = all_uploaders[1][uploaders]
        uploaders_emb2 = all_uploaders[2][uploaders]
        uploaders_emb3 = all_uploaders[3][uploaders]

        pos_uploaders_emb0 = all_uploaders[0][pos_uploaders]
        pos_uploaders_emb1 = all_uploaders[1][pos_uploaders]
        pos_uploaders_emb2 = all_uploaders[2][pos_uploaders]
        pos_uploaders_emb3 = all_uploaders[3][pos_uploaders]

        neg_uploaders_emb0 = all_uploaders[0][neg_uploaders]
        neg_uploaders_emb1 = all_uploaders[1][neg_uploaders]
        neg_uploaders_emb2 = all_uploaders[2][neg_uploaders]
        neg_uploaders_emb3 = all_uploaders[3][neg_uploaders]

        users_ua_emb_ego = self.embedding_user_ua(users)
        users_uv_emb_ego = self.embedding_user_uv(users)
        pos_emb_ego = self.embedding_video(pos_videos)
        neg_emb_ego = self.embedding_video(neg_videos)
        uploaders_emb_ego = self.embedding_uploader(uploaders)

        # 返回提取的嵌入向量
        # return users_emb, pos_emb, neg_emb, uploaders_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, uploaders_emb_ego
        return [users_emb0, users_emb1, users_emb2, users_emb3], pos_emb, neg_emb, \
            [uploaders_emb0, uploaders_emb1, uploaders_emb2, uploaders_emb3], \
            users_ua_emb_ego, users_uv_emb_ego, pos_emb_ego, neg_emb_ego, uploaders_emb_ego, \
            [pos_uploaders_emb0, pos_uploaders_emb1, pos_uploaders_emb2, pos_uploaders_emb3], \
            [neg_uploaders_emb0, neg_uploaders_emb1, neg_uploaders_emb2, neg_uploaders_emb3]

    def bpr_loss(self, users, pos, neg, uploaders):
        pos_uploader = self.get_video_uploader(pos)
        neg_uploader = self.get_video_uploader(neg)
        (users_emb, pos_emb, neg_emb, uploaders_emb, users_ua_emb_ego, users_uv_emb_ego, pos_emb_ego, neg_emb_ego,
         uploaders_emb_ego, pos_uploader_emb, neg_uploader_emb) = self.getEmbedding(users.long(), pos.long(),
                                                                                    neg.long(), uploaders.long(),
                                                                                    pos_uploader.long(),
                                                                                    neg_uploader.long())

        # 计算正则化损失
        # 正则化损失是用户和项目嵌入的L2范数的平方和
        # 这里的正则化损失是对所有用户的嵌入的平方和取平均
        reg_loss = (1 / 2) * (users_ua_emb_ego.norm(2).pow(2) +
                              users_uv_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2) + uploaders_emb_ego.norm(2).pow(2)) / float(users.numel())

        finall_users_emb = (users_emb[0] + users_emb[1] + users_emb[2] + users_emb[3]) / 4
        finall_pos_uploaders_emb = (pos_uploader_emb[0] + pos_uploader_emb[
            3]) / 2
        finall_neg_uploaders_emb = (neg_uploader_emb[0] + neg_uploader_emb[
            3]) / 2

        # 计算正样本的得分
        # 用户嵌入和正样本项目嵌入进行逐元素相乘，然后对每个用户的相乘结果求和
        pos_scores = torch.mul(finall_users_emb, (pos_emb + finall_pos_uploaders_emb) / 2)
        pos_scores = torch.sum(pos_scores, dim=1)

        # 计算负样本的得分
        # 用户嵌入和负样本项目嵌入进行逐元素相乘，然后对每个用户的相乘结果求和
        neg_scores = torch.mul(finall_users_emb, (neg_emb + finall_neg_uploaders_emb) / 2)
        neg_scores = torch.sum(neg_scores, dim=1)

        # 计算BPR损失
        # 使用softplus函数计算负样本得分和正样本得分的差异，并取平均值
        loss = torch.mean(F.softplus(neg_scores - pos_scores) + 10e-8)

        cl_loss = self.calc_crosscl_loss(users_emb, uploaders_emb)

        # 返回总损失，包括BPR损失和正则化损失
        # all_loss = loss + self.l2_w * reg_loss + self.cl_w * cl_loss
        return loss, reg_loss, cl_loss
        # print("loss:{}\treg_loss:{}\tcl_loss:{}\tall_loss:{}".format(loss, reg_loss, cl_loss, all_loss))
        # return all_loss

    def get_video_uploader(self, videos):
        uploader_ids = []
        # 遍历videos张量
        for video_id in videos:
            # 查找video_id对应的uploader_id
            uploader_id = self.video_uploader_dict.get(video_id.item(), -1)  # 如果没有找到，返回-1
            uploader_ids.append(uploader_id)

        # 将uploader_ids列表转换为张量
        return torch.tensor(uploader_ids)

    def calc_crosscl_loss(self, users_emb, uploader_emb):
        # 提取用户嵌入
        user_emb0 = users_emb[0]
        user_emb1 = users_emb[1]
        user_emb2 = users_emb[2]
        user_emb3 = users_emb[3]
        #
        # 提取视频博主嵌入
        uploader_emb0 = uploader_emb[0]
        uploader_emb1 = uploader_emb[1]
        uploader_emb2 = uploader_emb[2]
        uploader_emb3 = uploader_emb[3]

        # 通过投影层处理用户嵌入
        user_emb0 = self.userProjection(user_emb0)
        user_emb1 = self.userProjection(user_emb1)
        user_emb2 = self.userProjection(user_emb2)
        user_emb3 = self.userProjection(user_emb3)

        # 通过投影层处理视频博主嵌入
        uploader_emb0 = self.uploaderProjection(uploader_emb0)
        uploader_emb1 = self.uploaderProjection(uploader_emb1)
        uploader_emb2 = self.uploaderProjection(uploader_emb2)
        uploader_emb3 = self.uploaderProjection(uploader_emb3)

        # 对用户嵌入进行归一化
        normalize_emb_user0 = F.normalize(user_emb0, dim=1)
        normalize_emb_user1 = F.normalize(user_emb1, dim=1)
        normalize_emb_user2 = F.normalize(user_emb2, dim=1)
        normalize_emb_user3 = F.normalize(user_emb3, dim=1)

        # 对视频博主嵌入进行归一化
        normalize_emb_uploader0 = F.normalize(uploader_emb0, dim=1)
        normalize_emb_uploader1 = F.normalize(uploader_emb1, dim=1)
        normalize_emb_uploader2 = F.normalize(uploader_emb2, dim=1)
        normalize_emb_uploader3 = F.normalize(uploader_emb3, dim=1)
        normalize_emb_uploader3 = normalize_emb_uploader3.to(self.device)

        # 计算用户的正样本得分
        score_u1 = torch.sum(torch.mul(normalize_emb_user1, normalize_emb_user2), dim=1)
        score_u2 = torch.sum(torch.mul(normalize_emb_user0, normalize_emb_user3), dim=1)

        # 计算视频博主的正样本得分
        score_a1 = torch.sum(torch.mul(normalize_emb_uploader1, normalize_emb_uploader2), dim=1)
        score_a2 = torch.sum(torch.mul(normalize_emb_uploader0, normalize_emb_uploader3), dim=1)

        # 计算用户的总得分
        ttl_score_u1 = torch.matmul(normalize_emb_user1, normalize_emb_user2.T)
        ttl_score_u2 = torch.matmul(normalize_emb_user0, normalize_emb_user3.T)

        # 计算视频博主的总得分
        ttl_score_a1 = torch.matmul(normalize_emb_uploader1, normalize_emb_uploader2.T)
        ttl_score_a2 = torch.matmul(normalize_emb_uploader0, normalize_emb_uploader3.T)

        # 根据对比学习温度参数决定损失计算方式
        if self.cl_temp < 0.05:
            # 计算用户的InfoNCE损失
            ssl_logits_user1 = ttl_score_u1 - score_u1[:, None]  # [batch_size, num_users]
            ssl_logits_user2 = ttl_score_u2 - score_u2[:, None]  # [batch_size, num_users]

            # 计算视频博主的InfoNCE损失
            ssl_logits_uploader1 = ttl_score_a1 - score_a1[:, None]  # [batch_size, num_users]
            ssl_logits_uploader2 = ttl_score_a2 - score_a2[:, None]  # [batch_size, num_users]

            # 计算InfoNCE Loss
            clogits_user1 = torch.mean(torch.logsumexp(ssl_logits_user1 / self.cl_temp, dim=1))
            clogits_user2 = torch.mean(torch.logsumexp(ssl_logits_user2 / self.cl_temp, dim=1))
            clogits_uploader1 = torch.mean(torch.logsumexp(ssl_logits_uploader1 / self.cl_temp, dim=1))
            clogits_uploader2 = torch.mean(torch.logsumexp(ssl_logits_uploader2 / self.cl_temp, dim=1))
            # 取三者平均作为最终对比学习损失
            cl_loss = (torch.mean(clogits_user1) + torch.mean(clogits_user2) + torch.mean(clogits_uploader1)
                       + torch.mean(clogits_uploader2)) / 4
        else:
            # 当温度参数较大时，计算正样本得分的指数
            score_u1 = torch.exp(score_u1 / self.cl_temp)
            ttl_score_u1 = torch.sum(torch.exp(ttl_score_u1 / self.cl_temp), dim=1)
            score_u2 = torch.exp(score_u2 / self.cl_temp)
            ttl_score_u2 = torch.sum(torch.exp(ttl_score_u2 / self.cl_temp), dim=1)

            score_a1 = torch.exp(score_a1 / self.cl_temp)
            ttl_score_a1 = torch.sum(torch.exp(ttl_score_a1 / self.cl_temp), dim=1)
            score_a2 = torch.exp(score_a2 / self.cl_temp)
            ttl_score_a2 = torch.sum(torch.exp(ttl_score_a2 / self.cl_temp), dim=1)

            # 计算对比学习损失
            cl_loss = (torch.mean(torch.log(score_u1 / ttl_score_u1)) + torch.mean(
                torch.log(score_u2 / ttl_score_u2)) + torch.mean(
                torch.log(score_a1 / ttl_score_a1))
                       + torch.mean(
                        torch.log(score_a2 / ttl_score_a2))) / 4
        return -cl_loss

    def forward(self, users, items):
        # 调用 computer 方法，计算所有用户和所有项目的嵌入向量
        all_users, all_items = self.computer()

        # 从所有用户的嵌入向量中提取当前 batch 的用户嵌入向量
        users_emb = all_users[users]

        # 从所有项目的嵌入向量中提取当前 batch 的项目嵌入向量
        items_emb = all_items[items]

        # 逐元素相乘用户嵌入和项目嵌入向量
        inner_pro = torch.mul(users_emb, items_emb)

        # 对相乘结果的每一行进行求和，得到每个用户对每个项目的评分
        gamma = torch.sum(inner_pro, dim=1)

        # 返回评分
        return gamma


if __name__ == '__main__':
    dataset = Loader()
    model = IMP_GCN(dataset=dataset, config=config).to(device)
    user_ebmdding, uploader_embedding, video_embedding = model.compute()
    print(model.calc_crosscl_loss(user_ebmdding, uploader_embedding))
