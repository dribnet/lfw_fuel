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

"""
Labeled Faces in the Wild dataset, converted to fuel

Labeled Faces in the Wild is a database of face photographs
designed for studying the problem of unconstrained face recognition.

http://vis-www.cs.umass.edu/lfw/

This script currently packages the peopleDevTrain / peopleDevTest
split using the original version of the images.

"""


files = ['lfw.tgz', 'lfw-names.txt', 'peopleDevTest.txt', 'peopleDevTrain.txt']

########### Download section ##############


# this subparser hook is used for briq-download
def download_subparser(subparser):
    """
    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the babi_tasks command.
    """

    # this is a hack of the reqeusts library since many sitesgive a 403
    # against a python-requests user-agent.
    # import requests
    # requests.utils.default_user_agent = lambda: "Mozilla/5.0"

    urls = map(lambda s: 'http://vis-www.cs.umass.edu/lfw/' + s, files)

    subparser.set_defaults(
        func=default_downloader,
        urls=urls,
        filenames=files)


########### Convert section ##############

def loadImagesFromRow(tar, r):
    images = map(lambda n: "lfw/{0}/{0}_{1:04d}.jpg".format(r[0], n+1), range(int(r[1])))
    return map(lambda f:imread(tar.extractfile(f)), images)

def loadLabelsFromRow(labelslist, r):
    nameIndex = labelslist.index(r[0])
    return [nameIndex] * int(r[1])

# this should be equivalent to map(lambda r:loadImagesFromRow(tar, r), rows)
# but with a progress bar
def load_images(split, tar, rows):
    image_list = []
    progress_bar_context = progress_bar(
        name='{} images'.format(split), maxval=len(rows),
        prefix='Converting')
    with progress_bar_context as bar:
        for i, row in enumerate(rows):
            image_list.append(loadImagesFromRow(tar, row))
            bar.update(i)
    return image_list

@check_exists(required_files=files)
def convert_lfw(directory, output_directory, output_filename='lfw.hdf5'):

    # it will be faster to decompress this tar file all at once
    print("--> Converting tgz to tar")
    with gzip.open('lfw.tgz', 'rb') as f_in, open('lfw.tar', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print("--> Building test/train lists")
    # build lists, throwing away heading
    with open('peopleDevTrain.txt', 'rb') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
    with open('peopleDevTest.txt', 'rb') as csvfile:
        testrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

    print("--> Loading/converting images")
    tar = tarfile.open("lfw.tar")
    # extract all images in set
    train_images_nested = load_images("train", tar, trainrows)
    test_images_nested = load_images("test", tar, testrows)
    # test_images_nested = map(lambda r:loadImagesFromRow(tar, r), testrows)

    train_images_flat = np.array(reduce(lambda a,b: a+b, train_images_nested, []))
    test_images_flat  = np.array(reduce(lambda a,b: a+b, test_images_nested, []))

    print("--> Loading/converting labels")
    # now labels
    with open('lfw-names.txt', 'rb') as csvfile:
        labelrows = list(csv.reader(csvfile, delimiter='\t'))

    labelslist = map(lambda l:l[0], labelrows)
    labelslist.insert(0, 'Empty')

    # extract all labels in set
    train_labels_nested = map(lambda r:loadLabelsFromRow(labelslist, r), trainrows)
    test_labels_nested = map(lambda r:loadLabelsFromRow(labelslist, r), testrows)
    train_labels_flat = np.array(reduce(lambda a,b: a+b, train_labels_nested, []))
    test_labels_flat  = np.array(reduce(lambda a,b: a+b, test_labels_nested, []))

    print("--> Prepping hdf5 output file")
    # wrap for hdf5
    train_features = np.array([[r] for r in train_images_flat])
    test_features = np.array([[r] for r in test_images_flat])
    train_labels = np.array([[n] for n in train_labels_flat])
    test_labels = np.array([[n] for n in test_labels_flat])
    label_names = np.array([[n] for n in labelslist])

    output_path = os.path.join(output_directory, output_filename)

    # channel rearragement
    train_features_shaped = np.asarray([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2]] for f in train_features])
    test_features_shaped = np.asarray([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2]] for f in test_features])
    
    print("train shapes: ", train_features_shaped.shape, train_labels.shape)
    print("test shapes:  ", test_features_shaped.shape, test_labels.shape)
    print("target-names shape:  ", label_names.shape)
    
    print("--> Writing hdf5 output file")
    with h5py.File(output_path, mode="w") as h5file:
        data = (('train', 'features', train_features_shaped),
                ('train', 'targets', train_labels),
                ('test', 'features', test_features_shaped),
                ('test', 'targets', test_labels),
                ('target', 'names', label_names))
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['targets'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['names'].dims[i].label = label

    print("--> Done, removing tar file")
    os.remove("lfw.tar")
    return (output_path,)

def convert_subparser(subparser):
    subparser.set_defaults(func=convert_lfw)
