import os
import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.utils import get_argparser
import wandb

from src.All_Architecture import combinedModel
from src.encoders_ICA import NatureCNN, NatureOneCNN
from src.lstm_attn import subjLSTM
from src.slstm_attn_catalyst import LSTMTrainer
from src.bsnip_groups import GroupsDataset

# wandb.init(project="milc-bsnip2", entity="cedwards57")
parser = get_argparser()
args = parser.parse_args()
config = {}
config.update(vars(args))

data = GroupsDataset("./Data/bsnip2/bsnip2_labels.csv")
trainset, testset = train_test_split(
    data,
    test_size=.3,
    random_state=42
)
# train_loader = DataLoader(trainset, batch_size=20, shuffle=True)
# test_loader = DataLoader(testset, batch_size=20, shuffle=True)

wdb1 = "wandb_new"
wpath1 = os.path.join(os.getcwd(), wdb1)
dir = "PreTrainedEncoders/Milc/encoder.pt"
oldpath = wpath1 + "/PreTrainedEncoders/Milc"

if torch.cuda.is_available():
    cudaID = str(torch.cuda.current_device())
    device = torch.device("cuda:" + cudaID)
    # device = torch.device("cuda:" + str(args.cuda_id))
else:
    device = torch.device("cpu")

observation_shape = data.shape
encoder = NatureCNN(observation_shape[2], args)
lstm_model = subjLSTM(
    device,
    args.feature_size,
    args.lstm_size,
    num_layers=args.lstm_layers,
    freeze_embeddings=True,
    gain=current_gain,
)
complete_model = combinedModel(
    encoder,
    lstm_model,
    gain=current_gain,
    PT="milc",
    exp="UFPT",
    device=device,
    oldpath=oldpath,
)
config = {}
config.update(vars(args))
config["obs_space"] = observation_shape

trainer = LSTMTrainer(
    complete_model,
    config,
    device=device,
    tr_labels=tr_labels,
    val_labels=val_labels,
    test_labels=test_labels,
    wandb="wandb",
    trial=str(trial),
    gtrial=str(g_trial),
)

