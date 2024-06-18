import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import cust_mul
from dataloader import Loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMP_GCN(nn.Module):
    def __init__(self,
                 dataset,
                 latent_dim=200,
                 n_layers=6,
                 keep_prob=0.5,
                 groups=4,
                 device=torch.device('cuda'),
                 dropout_bool=False,
                 l2_w=1e-4,
                 cl_temp=0.1,
                 single=False):
        super(IMP_GCN, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout_bool = dropout_bool
        self.keep_prob = keep_prob
        self.Graph = dataset.getSparseGraph()
        self.num_users = dataset.n_user
        self.num_uploaders = dataset.n_uploaders
        self.num_videos = dataset.n_videos
        self.groups = groups
        self.device = device
        self.l2_w = l2_w
        self.cl_temp = cl_temp
        self.single = single
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_uploader = torch.nn.Embedding(self.num_uploaders, self.latent_dim)
        self.embedding_video = torch.nn.Embedding(self.num_videos, self.latent_dim)

        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc_g = torch.nn.Linear(self.latent_dim, self.groups)
        self.f = nn.Sigmoid()

        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_uploader.weight, std=0.1)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_uploader.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_video.weight, gain=1)
        # nn.init.xavier_uniform_(self.fc.weight, gain=1)
        # nn.init.xavier_uniform_(self.fc_g.weight, gain=1)

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
        users_emb = self.embedding_user.weight
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
        side_embed = torch.sparse.mm(g_droped[1], all_emb)

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
        i_one_hot = torch.ones(i_one_hot.shape).to(self.device)
        # 将用户和项目的one-hot嵌入表示拼接在一起并转置
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

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
        users_emb = self.embedding_user.weight
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

    def compute_video1_embedding(self):
        return 1
        pass

    def get_all_embedding(self):
        user_0, uploader_0 = self.computer_ua(self.dropout_bool)
        user_1, uploader_1 = self.computer_ua(1)
        user_2, uploader_2 = self.computer_ua(1)
        user_3, video_0 = self.computer_uv(self.dropout_bool)
        video_1 = self.compute_video1_embedding()

        return [user_0, user_1, user_2, user_3], [uploader_0, uploader_1, uploader_2], [video_0, video_1]

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        # 调用computer方法，计算所有用户和项目的嵌入
        all_users, all_items = self.computer()

        # 提取指定用户的嵌入
        users_emb = all_users[users]
        # 提取指定正样本项目的嵌入
        pos_emb = all_items[pos_items]
        # 提取指定负样本项目的嵌入
        neg_emb = all_items[neg_items]

        # 获取指定用户的初始嵌入（自有嵌入）
        users_emb_ego = self.embedding_user(users)
        # 获取指定正样本项目的初始嵌入（自有嵌入）
        pos_emb_ego = self.embedding_item(pos_items)
        # 获取指定负样本项目的初始嵌入（自有嵌入）
        neg_emb_ego = self.embedding_item(neg_items)

        # 返回提取的嵌入向量
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        # 获取用户、正样本项目和负样本项目的嵌入向量
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # 计算正则化损失
        # 正则化损失是用户和项目嵌入的L2范数的平方和
        # 这里的正则化损失是对所有用户的嵌入的平方和取平均
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        # 计算正样本的得分
        # 用户嵌入和正样本项目嵌入进行逐元素相乘，然后对每个用户的相乘结果求和
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        # 计算负样本的得分
        # 用户嵌入和负样本项目嵌入进行逐元素相乘，然后对每个用户的相乘结果求和
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # 计算BPR损失
        # 使用softplus函数计算负样本得分和正样本得分的差异，并取平均值
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # 返回总损失，包括BPR损失和正则化损失
        return loss + self.l2_w * reg_loss

    def calc_crosscl_loss(self, users_emb, uploader_emb):
        # 提取用户嵌入
        user_emb0 = users_emb[0]
        user_emb1 = users_emb[1]
        user_emb2 = users_emb[2]
        user_emb3 = users_emb[3]

        # 提取视频博主嵌入
        uploader_emb1 = uploader_emb[1]
        uploader_emb2 = uploader_emb[2]

        # 对用户嵌入进行归一化
        normalize_emb_user0 = F.normalize(user_emb0, dim=1)
        normalize_emb_user1 = F.normalize(user_emb1, dim=1)
        normalize_emb_user2 = F.normalize(user_emb2, dim=1)
        normalize_emb_user3 = F.normalize(user_emb3, dim=1)

        # 对视频博主嵌入进行归一化
        normalize_emb_uploader1 = F.normalize(uploader_emb1, dim=1)
        normalize_emb_uploader2 = F.normalize(uploader_emb2, dim=1)

        # 计算用户的正样本得分
        score_u1 = torch.sum(torch.mul(normalize_emb_user1, normalize_emb_user2), dim=1)
        score_u2 = torch.sum(torch.mul(normalize_emb_user0, normalize_emb_user3), dim=1)

        # 计算视频博主的正样本得分
        score_a = torch.sum(torch.mul(normalize_emb_uploader1, normalize_emb_uploader2), dim=1)

        # 计算用户的总得分
        ttl_score_u1 = torch.matmul(normalize_emb_user1, normalize_emb_user2.T)
        ttl_score_u2 = torch.matmul(normalize_emb_user0, normalize_emb_user3.T)

        # 计算视频博主的总得分
        ttl_score_a = torch.matmul(normalize_emb_uploader1, normalize_emb_uploader2.T)

        # 根据对比学习温度参数决定损失计算方式
        if self.cl_temp < 0.05:
            # 计算用户的InfoNCE损失
            ssl_logits_user1 = ttl_score_u1 - score_u1[:, None]  # [batch_size, num_users]
            ssl_logits_user2 = ttl_score_u2 - score_u2[:, None]  # [batch_size, num_users]

            # 计算视频博主的InfoNCE损失
            ssl_logits_uploader = ttl_score_a - score_a[:, None]  # [batch_size, num_users]

            # 计算InfoNCE Loss
            clogits_user1 = torch.mean(torch.logsumexp(ssl_logits_user1 / self.cl_temp, dim=1))
            clogits_user2 = torch.mean(torch.logsumexp(ssl_logits_user2 / self.cl_temp, dim=1))
            clogits_uploader = torch.mean(torch.logsumexp(ssl_logits_uploader / self.cl_temp, dim=1))
            # 取三者平均作为最终对比学习损失
            cl_loss = (torch.mean(clogits_user1) + torch.mean(clogits_user2) + torch.mean(clogits_uploader)) / 3
        else:
            # 当温度参数较大时，计算正样本得分的指数
            score_u1 = torch.exp(score_u1 / self.cl_temp)
            ttl_score_u1 = torch.sum(torch.exp(ttl_score_u1 / self.cl_temp), dim=1)
            score_u2 = torch.exp(score_u2 / self.cl_temp)
            ttl_score_u2 = torch.sum(torch.exp(ttl_score_u2 / self.cl_temp), dim=1)

            score_a = torch.exp(score_a / self.cl_temp)
            ttl_score_a = torch.sum(torch.exp(ttl_score_a / self.cl_temp), dim=1)

            # 计算对比学习损失
            cl_loss = (torch.mean(torch.log(score_u1 / ttl_score_u1)) + torch.mean(
                torch.log(score_u2 / ttl_score_u2)) + torch.mean(
                torch.log(score_a / ttl_score_a))) / 3
        return cl_loss

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
    model = IMP_GCN(dataset=dataset, device=device).to(device)
    user_ebmdding, uploader_embedding, video_embedding = model.get_all_embedding()
    # print(model.calc_crosscl_loss(user_ebmdding, uploader_embedding))
