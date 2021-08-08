import pandas as pd
import numpy as np
import os
import torch
import imagehash
from tqdm.auto import  tqdm
from PIL import Image

if __name__=="__main__":
    funcs = [
            imagehash.average_hash,
            imagehash.phash,
            imagehash.dhash,
            imagehash.whash,
        ]
    image_paths = []
    hashes = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/BIMCV/images'):
        for file in files:
            if '.jpg' in file or '.png' in file:
                image_paths.append(os.path.join(rdir, file))
    for rdir, _, files in os.walk('../../dataset/external_dataset/MIDRC-RICORD/images'):
        for file in files:
            if '.jpg' in file or '.png' in file:
                image_paths.append(os.path.join(rdir, file))
    for rdir, _, files in os.walk('../../dataset/siim-covid19-detection/images'):
        for file in files:
            if '.jpg' in file or '.png' in file:
                image_paths.append(os.path.join(rdir, file))
    
    for path in tqdm(image_paths):
        image = Image.open(path)
        image = image.resize((1024, 1024), Image.ANTIALIAS)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))
    sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy()/256 for i in range(hashes_all.shape[0])])
    indices1 = np.where(sims > 0.85)

    indices2 = np.where(indices1[0] != indices1[1])
    image_paths1 = [image_paths[i] for i in indices1[0][indices2]]
    image_paths2 = [image_paths[i] for i in indices1[1][indices2]]
    dups = {tuple(sorted([image_id1,image_id2])):True for image_id1, image_id2 in zip(image_paths1, image_paths2)}
    print('found %d duplicates' % len(dups))