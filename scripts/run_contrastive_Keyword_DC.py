import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import sys
import os
from src.dim_baseline import DIMTrainer
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser
from src.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.fine_tuning import FineTuning
import wandb
import pandas as pd
import datetime
from aari.episodes import get_episodes
import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gc
from pydub import AudioSegment
from IPython.display import Audio
from scipy import stats
import random



def train_encoder(args):
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1
    wdb='wandb'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")


    Adata = []
    ccount = 0
    p = 0
    tc = 20000
    subjects = 193
    sample_x = 128
    data = np.zeros((subjects, sample_x, tc))
    readerID=""
    start_path = '/Users/umahmood1/Desktop/AudioOutput/LibreSpeechData/Train_LibriSpeech/train-clean'
    for dirpath, dirnames, filenames in os.walk(start_path):
        dirnames.sort(key=int)
        for f in sorted(filenames, reverse=True):
            if (".h5" in f):
                nam = f.split('-')
                if (nam[0]!=readerID):
                    readerID = nam[0]
                    inputfilename = os.path.join(dirpath, f)
                    hf = h5py.File(inputfilename, 'r')
                    mdata = hf.get('mel_spec')
                    mdata = np.array(mdata)
                    if (mdata.shape[1]>=tc):
                        data[p, :, :] = mdata[:, 0:tc]
                        p=p+1
                # Adata.append(mdata)
                        ccount = ccount + 1
                        if ccount % 100 == 0:
                            print(ccount)
                break

    data2 = data.reshape(subjects, sample_x * tc)
    hf = h5py.File('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/libreSpeech_AllData_New.h5', 'w')
    hf.create_dataset('libreSpeech_dataset_New', data=data2)
    hf.close()
    print(ccount)
    # hf = h5py.File('../libreSpeech_AllData.h5', 'r')
    # data2 = hf.get('libreSpeech_dataset')
    # data2 = np.array(data2)
    # data2 = data2.reshape(subjects, sample_x, tc)
    ppp=0

    # for p in range(subjects):
    #     ppp=0
    #     for q in range(sample_x):
    #         for r in range(tc):
    #             diff = data[p, q, r] - data2[p, q, r];
    #             if (p%10==0 and ppp==0):
    #                 print(p)
    #                 ppp=1
    #             if(diff !=0):
    #                 print('Not Equal ...................')

    return

    # tsDim = np.zeros(584)
    # for a in range(584):
    #     tsDim[a] = len(Adata[a][0])


    # a=np.zeros((5,5))
    # np.savetxt('/Users/umahmood1/Desktop/a/c.csv, mfcc, delimiter=",")
    # filename = '/Users/umahmood1/Desktop/LibreSpeechData/Train_LibriSpeech/train-clean/19/227/19-227-0000.flac'
    # BG_y, BG_sr = librosa.load(filename)
    # librosa.output.write_wav('/Users/umahmood1/Desktop/a/outputAudioBG.wav', y=BG_y, sr=BG_sr)




    # For ICA TC Data

    speech_data = np.zeros((50000, 500, 1000))
    # modifieddata = np.zeros((823, 100, 1100))
    finalData = np.zeros((311, 13, 100, 20))
    finalData2 = np.zeros((311, 13, 53, 20))



    # filename = '/Users/umahmood1/Downloads/speech_commands/happy/01d22d03_nohash_0.wav'
    # filename = '/Users/umahmood1/Downloads/speech_commands/happy/4640-19187-0000.flac'
    # filename = '/Users/umahmood1/Desktop/LibreSpeechData/Train_LibriSpeech/train-clean/19/227/19-227-0000.flac'

    BG_noise_index = 0
    filename = '/Users/umahmood1/Desktop//AudioInput/LibreSpeechData/CF_Background_Noise.wav'
    BG_y, BG_sr = librosa.load(filename)


    in_train = False
    in_test = False
    data_dir = ""
    counter=0
    csv_dir = ""
    hfname=""
    hfnameT=""
    complete_data = []
    data_index = 0
    indices_arr = np.full((200, 1), -1)
    start_path = '/Users/umahmood1/Desktop/AudioInput/LibreSpeechData/Train_LibriSpeech'
    minx = 10000
    miny = 10000
    for dirpath, dirnames, filenames in os.walk(start_path):
        if("train-clean" in dirpath and in_train == False):
            data_dir="Train"
            counter=0;
            in_train = True;
            print(os.getcwd())
            csv_dir = '../LibreSpeechData/Train_CSVqw/'
        elif("test-clean" in dirpath and in_test == False):
            data_dir="Test"
            counter = 0;
            in_test = True;
            csv_dir = '../LibreSpeechData/Test_CSVqw/'
        # filenames = filenames.sort()
        for f in sorted(filenames):
            if (".flac" in f):
                outputpath = dirpath.replace('AudioInput', 'AudioOutput')

                if not os.path.exists(outputpath):
                    os.makedirs(outputpath)
                    if hfname !="":
                        hf = h5py.File(hfname, 'w')
                        hf.create_dataset('mel_spec', data=np.asarray(complete_data, dtype=np.float32))
                        hf.close()

                        np.savetxt(hfnameT, indices_arr, delimiter=",")
                        complete_data = []
                        indices_arr = np.full((200, 1), -1)
                        data_index=0
                    nam=f.split('-')
                    hfname = os.path.join(outputpath, nam[0]+'-'+nam[1]+'_AllData.h5')
                    hfnameT = os.path.join(outputpath, nam[0] + '-' + nam[1] + 'T'+'_Indices.csv')


                inputfilename = os.path.join(dirpath, f)

                fname = f.split('.')[0]
                output_filename = os.path.join(outputpath, fname)
                y, sr = librosa.load(inputfilename)

                if (BG_noise_index + y.shape[0] < BG_y.shape[0]):
                    ybg = BG_y[BG_noise_index:BG_noise_index + y.shape[0]]
                    BG_noise_index = BG_noise_index + y.shape[0]
                else:
                    BG_noise_index = 0
                    ybg = BG_y[BG_noise_index:BG_noise_index + y.shape[0]]
                    BG_noise_index = BG_noise_index + y.shape[0]

                y = stats.zscore(y, axis=0)
                ybg = stats.zscore(ybg, axis=0)

                r = random.randint(1, 4)
                if (r >1 and r<4):
                    alpha = 0.5
                    beta = 0.5
                elif (r==1):
                    alpha = 0.3
                    beta = 0.7
                elif (r==4):
                    alpha = 0.7
                    beta = 0.3
                combined_y = ((alpha * y) + (beta * ybg))

                librosa.output.write_wav(output_filename+'.wav', y=combined_y, sr=sr)
                m_spec = librosa.feature.melspectrogram(y=combined_y, sr=sr)
                x = m_spec.shape[0]
                y = m_spec.shape[1]

                if(x < minx):
                    minx=x
                if(y < miny):
                    miny=y
                # speech_data[counter, :x, :y] = m_spec[:,:]
                csvname = os.path.join(outputpath ,  fname + ".csv")
                counter = counter + 1

                if (counter % 100 == 0):
                    print(counter)
                    # gc.enable()
                if(len(complete_data)==0):
                    complete_data=m_spec
                else:
                    complete_data = np.concatenate((complete_data,m_spec),axis=1)
                indices_arr[data_index,0] = m_spec.shape[1]
                data_index=data_index+1
                # complete_dataT.append(m_spec)
                # np.savetxt(output_filename+'.csv', m_spec, delimiter=",")

    # plt.figure(figsize=(10, 4))
    # m_spec_dB = librosa.power_to_db(m_spec, ref=np.max)
    # librosa.display.specshow(m_spec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-frequency spectrogram')
    # plt.tight_layout()
    # plt.show()
    print(counter)
    print(minx)
    print(miny)
    return
    hf = h5py.File('../FBIRN_AllData.h5', 'r')
    data2 = hf.get('FBIRN_dataset')
    data2 = np.array(data2)
    data2 = data2.reshape(311, 100, 140)
    data = data2

    for i in range(311):
        for j in range(13):
            finalData[i, j, :, :] = data[i, :, (j * 10):(j * 10) + 20]

    print(finalData.shape)
    filename = 'index_array_labelled_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(311)

    filename = 'correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices, :]
    # print(c_indices)

    # index_array = torch.randperm(823)
    # index_array = torch.randperm(823)
    # tr_index = index_array[0:500]
    # val_index = index_array[0:500]
    # test_index = index_array[0:500]

    ID = args.script_ID - 1
    ID = ID * 100
    # val_indexs = ID-1;
    # val_indexe = ID+200

    test_indexs = ID
    test_indexe = ID + 100

    tr_index1 = index_array[0:test_indexs]
    tr_index2 = index_array[test_indexe:311]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index = index_array[test_indexs:test_indexe]
    # print(tr_index)
    tr_eps = finalData2[tr_index, :, :, :]
    val_eps = finalData2[500:700, :, :, :]
    test_eps = finalData2[test_index, :, :, :]

    tr_eps = torch.from_numpy(tr_eps).float()
    val_eps = torch.from_numpy(val_eps).float()
    test_eps = torch.from_numpy(test_eps).float()
    input_data = torch.rand(20, 10, 1, 160, 210)
    # tr_eps = torch.rand(20, 10, 1, 160, 210)
    # val_eps = torch.rand(20, 10, 1, 160, 210)
    # observation_shape = tr_eps[0][0].shape

    tr_eps.to(device)
    val_eps.to(device)
    test_eps.to(device)

    print("index_arrayshape", index_array.shape)
    print("trainershape", tr_eps.shape)
    print("valshape", val_eps.shape)
    print("testshape", test_eps.shape)
    initial_n_channels = 1
    print('ID = ', args.script_ID)

    observation_shape = finalData2.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)

    elif args.encoder_type == "NatureOne":
        encoder = NatureOneCNN(observation_shape[2], args)
    dir = 'run-2019-09-1018:25:39/encoder.pt'
    ep = os.path.join(wpath, dir)
    model_dict = torch.load(ep, map_location=device)  # with good components
    encoder.load_state_dict(model_dict)
    encoder.to(device)
    # torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    # print("trainershape", os.path.join(wandb.run.dir, config['env_name'] + '.pt'))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'spatial-appo':
        trainer = SpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "fine-tuning":
        trainer = FineTuning(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    # print("valshape", val_eps.shape)
    trainer.train(tr_eps, val_eps, test_eps)

    # return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    # wandb.init()
    #wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    # wandb.config.update(config)
    train_encoder(args)
