########## author: vasigiurgi #############


import os
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import rotate

def catg_img_path (dir, ext, catg):
    # Path to files in each category: um,umm, and uu
    img_path=sorted(
    [
        os.path.join(dir, fname)
        for fname in os.listdir(dir)
        if fname.endswith(ext) and fname.startswith(catg)
    ]
)
    return img_path



def split_check(paths_list):
    #split a path in to file keys for preparing split checking    
    file_names=[]
    for item in  paths_list:
        file_key=os.path.split(item)[1].split('.')[0]
        f_split=file_key.split('_')
        if 'road' in f_split:
            file_key= f_split[0]+ '_' + f_split[2]
        else:
            pass
        file_names.append(file_key)
    return file_names



def write_path(out_dir,paths_list,file_name):
    # Write the validation paths to text file
    path_name=os.path.join(out_dir,file_name)
    with open(path_name, 'w') as f:
      # write elements of list
      for items in paths_list:
            f.write('%s\n' %items)
    # close the file
    f.close()
    return 0

class kittiroad_camera(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays) for RGB images only."""

    def __init__(self, batch_size, img_size, input_img_paths_rgb, val=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths_rgb = input_img_paths_rgb
        self.val = val

    def __len__(self):
        return len(self.input_img_paths_rgb) // self.batch_size

    def __getitem__(self, idx):
        """Returns a batch of RGB images."""
        i = idx * self.batch_size
        batch_input_img_paths_rgb = self.input_img_paths_rgb[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        angle = random.uniform(-20, 20)
        
        for j, path in enumerate(batch_input_img_paths_rgb):
            img = load_img(path)
            img = img_to_array(img)
            h, w, _ = img.shape
            dh = int((self.img_size[0] - h) / 2)
            dw = int((self.img_size[1] - w) / 2)
            zero_padded_img = np.zeros(self.img_size + (3,), dtype="float32")
            zero_padded_img[dh:dh+h, dw:dw+w, :] = img / 255.0  # Normalize the image
            
            if not self.val:  # Training mode
                zero_padded_img = rotate(zero_padded_img, angle, mode='reflect')
                
            x[j] = zero_padded_img
            
        return x

