import torch
import numpy as np
import world


def UniformSample(dataset):
    users = np.random.randint(0, dataset.n_user, dataset.traindataSize)
    uploaders = np.random.randint(0, dataset.n_uploader, dataset.traindataSize)
    allPos = dataset.allPos
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        positem = posForUser[np.random.randint(0, len(posForUser))]
        negitem = np.random.randint(0, dataset.n_video)
        uploader = np.random.randint(0, dataset.n_uploader)
        while negitem in posForUser:
            negitem = np.random.randint(0, dataset.n_video)

        S.append([user, positem, negitem, uploader])

    return np.array(S)


def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# def generate_batches(tensors, batch_size):
#     for i in range(0, len(tensors), batch_size):
#         yield tensors[i:i + batch_size]


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, videos in enumerate(test_data):
        length = k if k <= len(videos) else len(videos)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos videos. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}
