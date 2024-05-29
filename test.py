import numpy as np
import nibabel as nib
data_path='/home/nasia/Documents/BRATS/data/labels.nii.gz'
data = nib.load(data_path)
np_data=data.get_fdata()
valuesn=np.unique(np_data)
print(valuesn)