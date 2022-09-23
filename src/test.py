from bsnip_datasets import RawDataset
from torch.utils.data import DataLoader
import torch

def my_collate_fn(batch):
    print(len(batch))

data = RawDataset(update_csv=False)
dl = DataLoader(
    data,
    batch_size=2,
    num_workers=0,
    shuffle=True,
    collate_fn=None
)

for i, (features, label) in enumerate(dl):
    if type(label) is not torch.Tensor:
        print(label)