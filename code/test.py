from utils import UniformSample
from dataloader import Loader
import world

if __name__ == '__main__':
    dataset = Loader(config=world.config, path='../data/wechat')
    print(UniformSample(dataset))
