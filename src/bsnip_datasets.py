import torch
import torch.nn
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import glob
from pathlib import Path
import numpy as np

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


class RawDataset(Dataset):
  def __init__(self, update_csv=False, raw_dir="./Data/bsnip2/raw/Raw_Data", processing="raw", select_groups={"SZ", "NC"}):
    '''raw_dir and select_groups will be ignored unless update_csv=True.'''
    super(RawDataset, self).__init__()
    processing_dict = {
      "raw": "", # min shape (222, 30, 64, 64)
      "nsp": "NSp", # Normalized & spatial warped; min shape (200, 52, 63, 53)
      "smnsp": "SmNSp" # Smoothed, normalized, & spatial warped; min shape (200, 52, 63, 53)
    }
    self.select_groups = select_groups
    self.raw_dir = raw_dir
    self.processing = processing.lower()
    self.files = list(glob.glob(f"{raw_dir}/*/*/ses_01/func1/{processing_dict[self.processing]}rest.nii"))
    if update_csv: self.__update_csv()
    self.df = pd.read_csv("./Data/bsnip2/bsnip2_raw_labels.csv", dtype={"subject_id": int, "filename": str, "group": int})
    self.shape = self.df.shape
    self.nifti_files = self.df['filename']
    self.labels = self.df['group']
    
  def __getitem__(self, k):
    label = self.labels[k]
    if isinstance(k, int):
      img = nib.load(self.nifti_files[k])
      img = img.slicer[:,:,:,:200]
      img = img.get_fdata().T
    
    else:
      img = []
      for i in self.nifti_files[k]:
        new = nib.load(i)
        new = new.slicer[:,:,:,:200]
        new = new.get_fdata().T
        img.append(new)
      img = np.stack(img)
      label = label.to_numpy()
    return img, label
  
  def __len__(self):
    return self.df.shape[0]
  
  def __update_csv(self):
    df = pd.DataFrame(columns={"subject_id": int, "filename": str, "group": int})
    groups_list = list(set(self.select_groups))
    groups_map = dict(zip(groups_list, range(len(groups_list))))
    omitted_subjects = {22281, 22111, 23333, 23520, 23083, 21193, 24077} # unusually tiny dimensions for raw data
    info = pd.read_csv("./Data/bsnip2/bsnip2_ad_preliminary_20201221.csv")
    for file in self.files:
      try:
        subject_id = int(Path(file).parents[2].name)
        csv_row = info.loc[info["subject_id"] == subject_id].to_dict("records")[0]
        if csv_row["group"] in self.select_groups and subject_id not in omitted_subjects:
          new_df_row = {
            "subject_id": subject_id,
            "filename": Path(file).absolute(),
            "group": groups_map[csv_row["group"]]
          }
          df = df.append(new_df_row, ignore_index=True)
      except (IndexError, ValueError):
        # IndexError: subject_id not found in the info CSV
        # ValueError: "PILOT" subject instead of an ID number; failed int conversion
        pass
    df.to_csv("./Data/bsnip2/bsnip2_raw_labels.csv")
    
# GA: 243
# rest = raw
# Sp = spatial warping
# SmNSp = smoothed & normalized
# NSp = normalized & spatial warped
# use glob for file searching
# prediction of groups
# ask about time direction generation/label code
# dimensions: batch, x, y, z, channels
# previously: batch, x, y, channels


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