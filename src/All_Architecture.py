import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as tn

class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(
        self,
        encoder,
        lstm,
        gain=0.1,
        PT="",
        exp="UFPT",
        device="cuda",
        oldpath="",
        complete_arc=False,
        num_classes=2
    ):

        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.gain = gain
        self.PT = PT
        self.exp = exp
        self.device = device
        self.oldpath = oldpath
        self.complete_arc = complete_arc
        self.num_classes = num_classes
        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, self.lstm.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm.hidden_dim, self.num_classes),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoder.feature_size, self.lstm.hidden_dim),
        ).to(device)

        self.init_weight()
        if self.complete_arc == False:
            self.loadModels()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def loadModels(self):
        if self.PT in ["milc", "variable-attention", "two-loss-milc"]:
            if self.exp in ["UFPT", "FPT"]:
                print("in ufpt and fpt")
                if not self.complete_arc:
                    model_dict = torch.load(
                        os.path.join(self.oldpath, "encoder" + ".pt"),
                        map_location=self.device,
                    )
                    ###
                    # model_dict["main.6.weight"] = model_dict.pop("main.7.weight")
                    # model_dict["main.6.bias"] = model_dict.pop("main.7.bias")
                    # model_dict["main.9.weight"] = model_dict.pop("main.8.weight")
                    # model_dict["main.9.bias"] = model_dict.pop("main.8.bias")
                    ###
                    self.encoder.load_state_dict(model_dict)

                    model_dict = torch.load(
                        os.path.join(self.oldpath, "lstm" + ".pt"),
                        map_location=self.device,
                    )
                    self.lstm.load_state_dict(model_dict)
                    # self.model.lstm.to(self.device)

                    model_dict = torch.load(
                        os.path.join(self.oldpath, "attn" + ".pt"),
                        map_location=self.device,
                    )
                    self.attn.load_state_dict(model_dict)
                    # self.model.attn.to(self.device)
                else:
                    model_dict = torch.load(
                        os.path.join(self.oldpath, "best_full" + ".pth"),
                        map_location=self.device,
                    )
                    self.load_state_dict(model_dict)

    def get_attention(self, outputs):
        # print('in attention')
        weights_list = []
        for X in outputs:
            # t=time.time()
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            result = torch.stack(result)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)

        weights = weights.squeeze()
        # weights should have more than one dimension
        normalized_weights = F.softmax(weights, dim=0) #was dim=1
        # normalized_weights.shape:
        # oasis: [16,7]
        # bsnip resnet: [200] // expects 3D tensor
        # outputs.shape bsnip: [200, 1, 200]
        # print(normalized_weights.unsqueeze(1).shape, outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()
        # attn_applied: [200,1,200] -> [200, 200]
        logits = self.decoder(attn_applied)
        
        # print("attention decoder ", time.time() - t)
        return logits

    def forward(self, sx, mode="train"):
        # print("sx", sx.shape)
        # inputs = [self.encoder(x) for x in sx] # sliding over batches
        inputs = [self.encoder(torch.unsqueeze(sx[:, x, :, :, :],1).float()) for x in range(sx.shape[1])] # sliding over time
        inputs = [i[1] for i in inputs]
        inputs = torch.stack(inputs)
        inputs = torch.swapdims(inputs, 0, 1) # inputs.shape: [batch_size, time, encoding]
        outputs = self.lstm(inputs, mode) # outputs shape [200, 1, 200]
        logits = self.get_attention(outputs)
        return logits
