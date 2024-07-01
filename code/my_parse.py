import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go JuRanModel")
    parser.add_argument('--latent_dim', type=int, default=64,
                        help="the embedding size of JuRanModel")
    parser.add_argument('--bpr_batch', type=int, default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--n_layers', type=int, default=2,
                        help="the layer num of JuRanModel")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="the learning rate")
    parser.add_argument('--test_batch', type=int, default=128,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[10,20,50]",
                        help="@k test list")
    parser.add_argument('--load', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--keep_prob', type=float, default=0.3, help='graph dropout rate')
    parser.add_argument('--dropout_bool', type=bool, default=0, help='Whether to use dropout')
    parser.add_argument('--groups', type=int, default=3, help='uploader groups')
    parser.add_argument('--neg', type=int, default=1, help='')

    parser.add_argument('--projection_dim', type=int, default=16, help='')
    parser.add_argument('--projection_dropout', type=int, default=0.5, help='')

    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--dataset', type=str, default='wechat',
                        help="available datasets: [wechat, takatak]")
    parser.add_argument('--l2_w', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--cl_w', type=float, default=0.0005, help='')
    parser.add_argument('--cl_temp', type=float, default=0.5, help='')
    parser.add_argument('--single', type=bool, default=0, help='是否单层嵌入')

    return parser.parse_args()
