from dataloader import Loader
from model import IMP_GCN
import torch

dataset = Loader()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

model = IMP_GCN(dataset, latent_dim=256, n_layers=3, groups=3, dropout_bool=False, l2_w=0.0002, single=True).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
args = {"dataset": dataset,
         "model": model,
         "optimiser": optim,
         "filename": f"imp_gcn_s_d{300}_l{3}_reg{2}_lr{'001'}"}
