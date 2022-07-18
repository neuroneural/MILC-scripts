from collections import deque
import datetime
from itertools import chain
import os
import sys
import time

# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from src.All_Architecture import combinedModel
from src.encoders_ICA import NatureOneCNN
from src.lstm_attn import subjLSTM
from src.slstm_attn_catalyst import LSTMTrainer
from src.utils import get_argparser
from Data.oasis.ts import load_OASIS
import torch
import wandb
# from tensorboardX import SummaryWriter


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    start_time = time.time()
    # do stuff

    ID = args.script_ID - 1
    JobID = args.job_ID
    ID = 4
    print("ID = " + str(ID))
    print("exp = " + args.exp)
    print("pretraining = " + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + "_" + str(ID)

    Name = args.exp + "_FBIRN_" + args.pre_training
    dir = "run-" + d1 + d2 + Name
    dir = dir + "-" + str(ID)
    wdb = "wandb_new"
    output_path = "Output"
    opath = os.path.join(os.getcwd(), output_path)
    # path = os.path.join(wpath, dir)
    args.path = opath

    wdb1 = "wandb_new"
    wpath1 = os.path.join(os.getcwd(), wdb1)

    tfilename = str(JobID) + "outputFILE" + Name + str(ID)

    output_path = os.path.join(os.getcwd(), "Output")
    output_path = os.path.join(output_path, tfilename)

    ntrials = 1
    ngtrials = 10
    tr_sub = [15, 25, 50, 75, 100]

    # With 16 per sub val, 10 WS working, MILC default
    gain = [0.05, 0.05, 0.05, 0.05, 0.05]  # UFPT

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    args.gain = current_gain

    # COBRE like shape
    sample_y = 20
    tc = 140
    # OASIS like shape
    # sample_y = 26
    # tc = 156

    samples_per_subject = tc // sample_y
    # samples_per_subject = 13
    # samples_per_subject = int((tc - sample_y)+1)
    window_shift = sample_y
    # window_shift = 10

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print("device = ", device)
    # return
    # For ICA TC Data

    features, all_labels = load_OASIS()
    # have to truncate OASIS data, otherwise encoder has dimensionality mismatch
    # (see ./src/encoders_ICA.py: line 145)
    features = features[:, :, :tc]

    finalData = np.zeros(
        (features.shape[0], samples_per_subject, features.shape[1], sample_y)
    )

    #window shifting
    for i in range(features.shape[0]):
        for j in range(samples_per_subject):
            finalData[i, j, :, :] = features[
                i, :, (j * window_shift) : (j * window_shift) + sample_y
            ]
    finalData2 = torch.from_numpy(finalData).float()
    all_labels = torch.from_numpy(all_labels).int()

    print(finalData2.shape)
    print(all_labels.shape)

    results = torch.zeros(50, 4)

    for k in range(5):
        for trial in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            skf.get_n_splits(finalData2, all_labels)

            train_index, test_index = list(skf.split(features, all_labels))[k]

            X_train, X_test = finalData2[train_index], finalData2[test_index]
            y_train, y_test = all_labels[train_index], all_labels[test_index]

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=165,
                random_state=42 + trial,
                stratify=y_train,
            )

            # print(X_train.shape)
            # print(X_val.shape)
            # print(X_test.shape)
            # print(y_train.shape)
            # print(y_val.shape)
            # print(y_test.shape)

            print("k = ", k)
            print("trial = ", trial)

            g_trial = 1
            output_text_file = open(output_path, "a+")
            output_text_file.write("Trial = %d gTrial = %d\r\n" % (trial, g_trial))
            output_text_file.close()

            observation_shape = finalData2.shape
            print("Observation shape = ", observation_shape)
            # Observation shape =  torch.Size([823, 7, 53, 20])
            # Observation shape =  torch.Size([823, 6, 53, 26])

            dir = "PreTrainedEncoders/Milc/encoder.pt"
            args.oldpath = wpath1 + "/PreTrainedEncoders/Milc"

            path = os.path.join(wpath1, dir)

            encoder = NatureOneCNN(observation_shape[2], args)
            print(current_gain)
            lstm_model = subjLSTM(
                device,
                args.feature_size,
                args.lstm_size,
                num_layers=args.lstm_layers,
                freeze_embeddings=True,
                gain=current_gain,
            )

            model_dict = torch.load(path, map_location=device)  # with good components

            complete_model = combinedModel(
                encoder,
                lstm_model,
                gain=current_gain,
                PT=args.pre_training,
                exp=args.exp,
                device=device,
                oldpath=args.oldpath,
            )
            config = {}
            config.update(vars(args))
            config["obs_space"] = observation_shape  # weird hack

            if args.method == "sub-lstm":
                trainer = LSTMTrainer(
                    complete_model,
                    config,
                    device=device,
                    tr_labels=y_train,
                    val_labels=y_val,
                    test_labels=y_test,
                    wandb="wandb",
                    trial=str(trial),
                    gtrial=str(g_trial),
                )
            else:
                assert False, "method {} has no trainer".format(args.method)
            xindex = (k * 10) + trial
            (
                results[xindex][0],
                results[xindex][1],
                results[xindex][2],
                results[xindex][3],
            ) = trainer.pre_train(X_train, X_val, X_test)

    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, "test_results" + sID + ".csv")
    np.savetxt(tresult_csv, np_results, delimiter=",")
    elapsed = time.time() - start_time
    print("total time = ", elapsed)


if __name__ == "__main__":
    wandb.init(project="milc-oasis", entity="cedwards57")
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["pretraining-only"]
    config = {}
    config.update(vars(args))
    # for k in range(5):
    #     for triall in range(10):
    # train_encoder(args, k, triall)
    train_encoder(args)
