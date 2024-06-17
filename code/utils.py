import torch
import numpy as np


def UniformSample(dataset):
    users = np.random.randint(0, dataset.n_user, dataset.traindataSize)
    allPos = dataset.allPos
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        positem = posForUser[np.random.randint(0, len(posForUser))]
        negitem = np.random.randint(0, dataset.m_item)
        while negitem in posForUser:
            negitem = np.random.randint(0, dataset.m_item)

        S.append([user, positem, negitem])

    return np.array(S)


def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())
