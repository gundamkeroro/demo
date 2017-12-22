from PIL import Image
from os import listdir
import os
import subprocess
from os.path import isfile, join
import numpy as np
import pickle
from time import time
import sys
import h5py
from tqdm import tqdm

def prcessing():
    image_dir_s = 'test/'
    try:
        image_locs = [join(image_dir_s, f) for f in listdir(image_dir_s) if isfile(join(image_dir_s, f))]
    except:
        print "expected aligned images directory, see README"

    total_imgs = len(image_locs)
    print "found %i image in directory" % total_imgs


    def process_image(im):
        im = im.convert("RGB")
        new_size = [int(i / 1.3) for i in im.size]
        im.thumbnail(new_size, Image.ANTIALIAS)
        target = np.array(im)[4:-4, 4:-4, :]
        iput = np.array(im)[4:-4, 4:-4, :]
        return (iput, target)


    def proc_loc(loc):
        file = loc.split('/')[-1]
        os.chdir('test')
        subprocess.call('convert '  + file + ' -resize 178x220\! ' + file, shell=True)
        os.chdir('../')
        try:
            im = Image.open(loc)
            iput, target = process_image(im)
            return (iput, target)
        except KeyboardInterrupt:
            raise
        except:
            return None


    try:
        hf = h5py.File('faces_single_test.hdf5', 'r+')
    except:
        hf = h5py.File('faces_single_test.hdf5', 'w')
#target is not needed just for padding
    try:
        dset_t = hf.create_dataset("target", (1, 160, 128, 3),
                                   maxshape=(1e6, 160, 128, 3), chunks=(1, 160, 128, 3), compression="gzip")
    except:
        dset_t = hf['target']

    try:
        dset_i = hf.create_dataset("input", (1, 160, 128, 3),
                                   maxshape=(1e6, 160, 128, 3), chunks=(1, 160, 128, 3), compression="gzip")
    except:
        dset_i = hf['input']

    batch_size = 1
    num_iter = total_imgs / batch_size

    insert_point = 0
    print "STARTING PROCESSING IN BATCHES OF %i" % batch_size

    for i in tqdm(range(num_iter)):
        sys.stdout.flush()

        X_in = []
        X_ta = []

        a = time()
        locs = image_locs[i * batch_size: (i + 1) * batch_size]

        proc = []
        for i in range(len(locs)):
            pair = proc_loc(locs[i])
            proc.append(pair)

        for pair in proc:
            if pair is not None:
                iput, target = pair
                X_in.append(iput)
                X_ta.append(target)

        X_in = np.array(X_in)
        X_ta = np.array(X_ta)
        X_in = np.resize(X_in, (len(X_in), 160, 128, 3))
        X_ta = np.resize(X_ta, (len(X_ta), 160, 128, 3))

        try:
            dset_i.resize((insert_point + len(X_in), 160, 128, 3))
            dset_t.resize((insert_point + len(X_in), 160, 128, 3))

            dset_i[insert_point:insert_point + len(X_in)] = X_in
            dset_t[insert_point:insert_point + len(X_in)] = X_ta

            insert_point += len(X_in)
        except Exception as e:
            print "can't insert"

    subprocess.call("rm test/*", shell=True)

    hf.close()

if __name__ == '__main__':
    prcessing()