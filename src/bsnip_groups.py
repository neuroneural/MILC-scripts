import torch
import torch.nn
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import glob
device = "cuda"

class GroupsDataset(Dataset):
  def __init__(self, csv_location, MAX_TC=140):
    # max_TC sets max time
    super(GroupsDataset, self).__init__()
    self.df = pd.read_csv(csv_location)
    self.shape = self.df.shape
    self.nifti_files = self.df['filename']
    self.labels = self.df['group']
    self.MAX_TC = MAX_TC

  def __getitem__(self, k):
    print(self.nifti_files[57])
    print("#####")
    # load the nifti file from a given filename
    img = nib.load(self.nifti_files[k])
    img = img.slicer[:,:self.MAX_TC]
    # loads the corresponding label (integer)
    label = self.labels[k]

    img = img.get_fdata().T
    img = img.reshape(img.shape[0], img.shape[1], 1)

    return img, label
    
  
  def __len__(self):
    return len(self.labels)


### CODE THAT CREATED THE bsnip2_labels.csv FILE ###
# # grab & log correct subject IDs
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