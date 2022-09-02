import torch
import torch.nn
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import glob
import numpy as np
device = "cuda"

class GroupsDataset(Dataset):
  def __init__(self, csv_location, MAX_TC=200):
    # max_TC sets max time
    super(GroupsDataset, self).__init__()
    self.df = pd.read_csv(csv_location)
    self.shape = self.df.shape
    self.nifti_files = self.df['filename']
    self.labels = self.df['group']
    self.MAX_TC = MAX_TC

  def __getitem__(self, k):
    # loads the corresponding label (integer)
    label = self.labels[k]
    if isinstance(k,int):
      # load the nifti file from a given filename
      img = nib.load(self.nifti_files[k])
      img = img.slicer[:self.MAX_TC,:]
      img = img.get_fdata().T
      img = img.reshape((1, img.shape[0], img.shape[1]))

    else: # k is an ndarray. it has the same shape as what labels.shape outputs at the end.
      img = []
      for i in self.nifti_files[k]:
        new = nib.load(i)
        new = new.slicer[:self.MAX_TC,:]
        new = new.get_fdata().T
        new = new.reshape((1, new.shape[0], new.shape[1]))
        img.append(new)
      img = np.stack(img)
      # img = img.float()
      label = label.to_numpy()
      # label = label.float()
      # print(f"img.shape: {img.shape}")
      # print(f"label.shape: {label.shape}")
    return img, label
    
  def __len__(self):
    return len(self.labels)


### CODE THAT CREATED THE bsnip2_labels.csv FILE ###
# grab & log correct subject IDs
# data = pd.read_csv("./Data/bsnip2/bsnip2_ad_preliminary_20201221.csv")
# subjects = pd.read_csv("./Data/bsnip2/bsnip2_subjects.txt", header=None)
# subjects = list(subjects[0])
# groups = []
# # SZ, NC, SAD, BP, BPnon, OTH
# missing = []
# for i, subject_id in enumerate(subjects):
#   new = data.loc[data['subject_id'] == subject_id]
#   try:
#     if (list(new["group"])[0] == "OTH") or (list(new["group"])[0] == "BPnon") or (list(new["group"])[0] == "SAD") or (list(new["group"])[0] == "BP"):
#       missing.append(i)
#     else:
#       groups.append(list(new["group"])[0])
#   except IndexError:
#     missing.append(i)
# # missing_ids = [subjects[i] for i in missing]
# # print(missing_ids)

# # remove missing indeces from filenames list
# filenames = list(glob.glob("./Data/bsnip2/ts/BSNIP2_sub*_timecourses_ica_s1_.nii"))
# for i in reversed(missing):
#   filenames.remove(filenames[i])

# # map each group to a number, save all to the labels file
# groups_list = list(set(groups))
# groups_map = dict(zip(groups_list, range(len(groups_list))))
# # print(groups_map)
# dset = {"filename": filenames,"group": groups}
# df = pd.DataFrame(dset)
# df["group"] = df["group"].map(groups_map)
# df.to_csv("./Data/bsnip2/bsnip2_labels.csv")
### ###
# # specify num_classes in src/bsnip_slstm_attn_catalyst AccuracyCallback and in scripts/run_ICA_experiments_BSNIP2_catalyst