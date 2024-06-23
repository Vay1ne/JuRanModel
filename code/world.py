import os
from os.path import join
import torch
from enum import Enum
from my_parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "../../"

CODE_PATH = join(ROOT_PATH, '')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join('checkpoints')
import sys

sys.path.append(join(CODE_PATH, 'sources'))

# 这里注销掉创建checkpoints文件夹
# if not os.path.exists(FILE_PATH):
#     os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['wechat', 'takatak']
all_models = ['mf', 'lgn']

config['latent_dim'] = args.latent_dim
config['bpr_batch'] = args.bpr_batch
config['n_layers'] = args.n_layers
config['lr'] = args.lr
config['test_batch'] = args.test_batch
config['path'] = args.path
config['topks'] = args.topks
config['load'] = args.load
config['epochs'] = args.epochs
config['multicore'] = args.multicore
config['pretrain'] = args.pretrain
config['seed'] = args.seed
config['model'] = args.model
config['keep_prob'] = args.keep_prob
config['dropout_bool'] = args.dropout_bool
config['groups'] = args.groups
config['neg'] = args.neg
config['projection_dim'] = args.projection_dim
config['projection_dropout'] = args.projection_dropout
config['gpu_id'] = args.gpu_id
config['dataset'] = args.dataset
config['l2_w'] = args.l2_w
config['cl_w'] = args.cl_w
config['cl_temp'] = args.cl_temp
config['single'] = args.single

os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
# device = torch.device('cuda')
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)

# let pandas shut up
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
