import os
import sys
sys.path.append('.')

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from src.utils import get_argparser
import wandb

from src.All_Architecture import combinedModel
from src.encoders_ICA import NatureCNN
from src.lstm_attn import subjLSTM
from src.bsnip_slstm_attn_catalyst import BSNIPLSTMTrainer
from src.bsnip_groups import GroupsDataset

def train_encoder(args):
    data = GroupsDataset("./Data/bsnip2/bsnip2_labels.csv")
    fulltrain_idx, test_idx = train_test_split(
        list(range(len(data))),
        test_size=.3,
        random_state=42
    )

    train_idx, valid_idx = train_test_split(
        list(fulltrain_idx),
        test_size=.2,
        random_state=42
    )
    
    trainset = Subset(data, train_idx)
    testset = Subset(data, test_idx)
    validset = Subset(data, valid_idx)
    
    wdb1 = "wandb_new"
    wpath1 = os.path.join(os.getcwd(), wdb1)
    dir = "PreTrainedEncoders/Milc/encoder.pt"
    oldpath = wpath1 + "/PreTrainedEncoders/Milc"

    gain = [0.05, 0.05, 0.05, 0.05, 0.05]
    ID = 1
    current_gain = gain[ID]
    args.gain = current_gain

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")

    observation_shape = data.shape
    encoder = NatureCNN(observation_shape[1], args)
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
        complete_arc=True,
        num_classes=5
    )
    config = {}
    config.update(vars(args))
    config["obs_space"] = observation_shape

    # FBIRN - if args.fMRI_twoD -- if expected 4D input 
    # FBIRN 108-113 -- into get_item function


    trainer = BSNIPLSTMTrainer(
        complete_model,
        config,
        device=device,
        # tr_labels=tr_labels,
        # val_labels=val_labels,
        # test_labels=test_labels,
        trainset=trainset,
        testset=testset,
        validset=validset,
        wandb="wandb",
        trial="1",
        batch_size=32,
        # gtrial=str(g_trial),
    )
    trainer.train()

# test_acc, test_auc, test_loss = trainer.train()
if __name__ == "__main__":
    wandb.init(project="milc-bsnip2", entity="cedwards57", settings=wandb.Settings(start_method='fork'))
    parser = get_argparser()
    args = parser.parse_args()
    config = {}
    config.update(vars(args))
    train_encoder(args)
