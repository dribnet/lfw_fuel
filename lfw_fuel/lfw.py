from __future__ import absolute_import
from __future__ import print_function
import tarfile
import csv
import gzip
import shutil
import numpy as np
from scipy.misc import imread
import os
import h5py
import numpy

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar
from fuel.downloaders.base import default_downloader
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path

"""
Labeled Faces in the Wild dataset, converted to fuel

Labeled Faces in the Wild is a database of face photographs
designed for studying the problem of unconstrained face recognition.

http://vis-www.cs.umass.edu/lfw/

This project currently packages the pairsDevTrain / pairsDevTest
splits images into a fuel compatible dataset along with targets
that indicate whether the pairs are same or different. It supports
converting the original lfw dataset, as well as the funneled
and deepfunneled versions.

"""


files = ['lfw-names.txt', 'pairsDevTest.txt', 'pairsDevTrain.txt']
urlroot = 'http://vis-www.cs.umass.edu/lfw/'

########### Download section ##############

def resolve_filename(format):
    imfile = "lfw"
    if format == "funneled":
        imfile = "lfw-funneled"
    elif format == "deepfunneled":
        imfile = "lfw-deepfunneled"
    # could add superpixel here I guess..
    return imfile

# we need a wrapper around the default_downlaoder to resolve files
def downloader_wrapper(format, directory, **kwargs):
    # add the right format file to the download list
    files.insert(0, "{}.tgz".format(resolve_filename(format)))
    urls = list(map(lambda s: 'http://vis-www.cs.umass.edu/lfw/' + s, files))
    default_downloader(directory, urls=urls, filenames=files, **kwargs)

# this subparser hook is used for briq-download
def download_subparser(subparser):
    """
    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the babi_tasks command.
    """

    # optional format can be funneled, deepfunneled, etc
    subparser.add_argument(
        "--format", help="alternate format", type=str, default=None)

    urls = list(map(lambda s: 'http://vis-www.cs.umass.edu/lfw/' + s, files))

    return downloader_wrapper


########### Convert section ##############

def loadImage(tar, basename, name, number):
    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    return imread(tar.extractfile(filename))

def loadImagePairFromRow(tar, basename, r):
    if(len(r) == 3):
        # same
        return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[0], r[2])]
    else:
        # different
        return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[2], r[3])]

def loadLabelsFromRow(r):
    if(len(r) == 3):
        return 1
    else:
        return 0

# this should be equivalent to
#   np.array(map(lambda r:loadImagePairFromRow(tar, r), trainrows))
# but with a progress bar
def load_images(split, tar, basename, rows):
    image_list = []
    progress_bar_context = progress_bar(
        name='{} images'.format(split), maxval=len(rows),
        prefix='Converting')
    with progress_bar_context as bar:
        for i, row in enumerate(rows):
            image_list.append(loadImagePairFromRow(tar, basename, row))
            bar.update(i)
    return np.array(image_list)

@check_exists(required_files=files)
def convert_lfw(directory, basename, output_directory):
    tgz_filename = "{}.tgz".format(basename)
    tar_filename = "{}.tar".format(basename)
    output_filename = "{}.hdf5".format(basename)
    tar_subdir = "lfw_funneled" if basename == "lfw-funneled" else basename

    # it will be faster to decompress this tar file all at once
    print("--> Converting {} to tar".format(tgz_filename))
    with gzip.open(tgz_filename, 'rb') as f_in, open(tar_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    tar = tarfile.open(tar_filename)

    print("--> Building test/train lists")
    # build lists, throwing away heading
    with open('pairsDevTrain.txt', 'rb') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
    with open('pairsDevTest.txt', 'rb') as csvfile:
        testrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

    print("--> Converting")
    # extract all images in set
    train_images = load_images("train", tar, tar_subdir, trainrows)
    test_images  = load_images("test",  tar, tar_subdir, testrows)

    train_labels = np.array(list(map(lambda r:loadLabelsFromRow(r), trainrows)))
    test_labels = np.array(list(map(lambda r:loadLabelsFromRow(r), testrows)))

    train_features = np.array([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2], f[1,:,:,0], f[1,:,:,1], f[1,:,:,2]] for f in train_images])
    test_features  = np.array([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2], f[1,:,:,0], f[1,:,:,1], f[1,:,:,2]] for f in test_images])

    train_targets = np.array([[n] for n in train_labels])
    test_targets  = np.array([[n] for n in test_labels])

    print("train shapes: ", train_features.shape, train_targets.shape)
    print("test shapes:  ", test_features.shape, test_targets.shape)
    
    print("--> Writing hdf5 output file")
    output_path = os.path.join(output_directory, output_filename)
    with h5py.File(output_path, mode="w") as h5file:
        data = (('train', 'features', train_features),
                ('train', 'targets', train_targets),
                ('test', 'features', test_features),
                ('test', 'targets', test_targets))
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['targets'].dims[i].label = label

    print("--> Done, removing tar file")
    os.remove(tar_filename)
    return (output_path,)

# wrapper because check_exists is a decarator with directory first
def convert_lfw_wrapper(directory, format, **kwargs):
    # print("Got args: {}, {}, {}".format(directory, format, kwargs))
    basename = resolve_filename(format)
    files.insert(0, "{}.tgz".format(basename))
    return convert_lfw(directory, basename, **kwargs)

def convert_subparser(subparser):
    # optional format can be funneled, deepfunneled, etc
    subparser.add_argument(
        "--format", help="alternate format", type=str, default=None)
    return convert_lfw_wrapper


########### Fuel Dataset section ##############


class LFW(H5PYDataset):
    u"""LFW dataset.

    Labeled Faces in the Wild dataset.

    Labeled Faces in the Wild is a database of face photographs
    designed for studying the problem of unconstrained face recognition.

    http://vis-www.cs.umass.edu/lfw/

    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (50,000 examples) and the test
        set (10,000 examples).

    """
    url_dir = "https://archive.org/download/lfw_fuel/"
    filename = 'lfw.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(LFW, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


########### Kerosene Dataset section ##############


from kerosene.datasets.dataset import Dataset
from fuel.transformers.image import RandomFixedSizeCrop

class LFWDataset(Dataset):
    basename = "lfw"
    version = "0.1.0"
    url_dir = "https://archive.org/download/lfw_fuel/"
    class_for_filename_patch = LFW

    # def apply_transforms(self, datasets):
    #     return map(lambda x: RandomFixedSizeCrop(x, (32, 32),
    #                                  which_sources=('features',)), datasets)

    def build_data(self, sets, sources):
        return map(lambda s: LFW(which_sets=[s], sources=sources), sets)

def load_data(format=None, sets=None, sources=None, fuel_dir=False):
    dataset = LFWDataset()
    dataset.basename = resolve_filename(format)
    return dataset.load_data(sets, sources, fuel_dir);
