import torch
import torch.nn
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import glob
device = "cuda"


### CODE THAT CREATED THE bsnip2_labels.csv FILE ###
# data = pd.read_csv("./Data/bsnip2/bsnip2_ad_preliminary_20201221.csv")
# subjects = pd.read_csv("./Data/bsnip2/bsnip2_subjects.txt", header=None)
# subjects = list(subjects[0])
# groups = []
# missing = []
# for i, subject_id in enumerate(subjects):
#   new = data.loc[data['subject_id'] == subject_id]
#   try:
#     groups.append(list(new["group"])[0])
#   except IndexError:
#     missing.append(i)
#     continue

# filenames = list(glob.glob("./Data/bsnip2/ts/BSNIP2_sub*_timecourses_ica_s1_.nii"))
# for i in reversed(missing):
#   filenames.remove(filenames[i])

# groups_list = list(set(groups))
# groups_map = dict(zip(groups_list, range(len(groups_list))))
# # print(groups_map)
# dset = {"filename": filenames,"group": groups}
# df = pd.DataFrame(dset)
# df["group"] = df["group"].map(groups_map)
# df.to_csv("./Data/bsnip2/bsnip2_labels.csv")
###


class GroupsDataset(Dataset):
  def __init__(self, csv_location):
    super(GroupsDataset, self).__init__()
    self.df = pd.read_csv(csv_location)
    self.nifti_files = self.df['filename']
    self.labels = self.df['group']

  def __getitem__(self, k):
    # load the nifti file from a given filename
    img = nib.load(self.nifti_files[k])
    # loads the corresponding label (integer)
    label = self.labels[k]
    return img.get_fdata().T, label
    
  
  def __len__(self):
    return len(self.labels)

