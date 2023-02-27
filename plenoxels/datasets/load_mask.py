import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags

def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    #msk[msk == 1] = 255 # original

    return msk

def main(argv):
    subject = '387'
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    select_view = 0
    img_path_frames_views = annots['ims']
    img_paths = np.array([

        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in img_path_frames_views
    ])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]
    output_dir ='../../dataset/zju_mocap'
    output_path = os.path.join(output_dir, 
                               subject )
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')
    select_view = select_view

    for idx, ipath in enumerate(tqdm(img_paths)):
        #out_name = 'frame_{:06d}'.format(idx) #original
        out_name = 'r_{:03d}'.format(idx)

        img_path = os.path.join(subject_dir, ipath)
    
        # load image
        img = np.array(load_image(img_path))

        mask = get_mask(subject_dir, ipath)
        save_image(to_3ch_image(mask), os.pathrays_intersect_3d_bbox.join(out_mask_dir, out_name+'.png'))
            # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)