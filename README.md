# lfw_fuel: Labeled Faces in the Wild dataset, converted to fuel

Labeled Faces in the Wild is a database of face photographs
designed for studying the problem of unconstrained face recognition.

http://vis-www.cs.umass.edu/lfw/

This project currently packages the pairsDevTrain / pairsDevTest
splits images into a fuel compatible dataset along with targets
that indicate whether the pairs are same or different. It supports
converting the original lfw dataset, as well as the funneled
and deepfunneled versions.

## Show me

From the included example"

```python
from lfw_fuel import lfw
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = lfw.load_data(format="deepfunneled")
```

The features are currently stored in six channels - three for each
of the two RGB images to be compared.

Note that the images are `250x250` - which is quite large by most
CNN standards. These can be cropped and scaled before passing them
to the network as shown in the example.

## OK, what is this again?

Kerosene provides a collection of versioned, immutable, publicly available fuel-compatible datasets in hdf5 format along with a minimalist interface for Keras. Let's go through that quickly.

  * Semantic Versioning. Just like software. There will be bugs. There will be changes. We'll be ready.
  * Immutable. Once a version released to the wild, it is never rewritten.
  * Publicly available. Reproducable experiments depend on unencombered access.
  * fuel-compatible - borrows heavily and remains compatible with the [fuel data pipeline frameworkk](https://github.com/mila-udem/fuel)
  * hdf5 format. welcome to a saner world free of pickled python objects.
  * interface: As simple as possible. Automatic downloads and sensible defaults.

Kerosene itself includes wrappers only for the datasets that are built into the fuel libraries. But when
used as a dependency, it allows similar access to any third party fuel hdf5 file. As an example, see
the lfw_fuel repo which provides keras and blocks access to the Labeled Faces in the Wild dataset
in several formats.

## Installation

Currently depends on official Keras release and current version of fuel. Not yet installable via pip.

You probably know how to install Keras, so to get fuel just

```bash
pip install git+git://github.com/mila-udem/fuel.git@0653e5b

```

And then from this repo

```
python setup.py install
```

After that you should be able to run any of the examples in the examples folder

```bash
python ./examples/mnist.py
```

## What's included

Currently the six datasets are wrappers around those provided by fuel. Each has corresponding
example in the examples directory which is meant to be a high performance representative use of that
dataset.

There's also small wrapper scripts `kero-download` and `kero-convert`, which are used to run `fuel-download`
and `fuel-convert` on datasets that are not part of the fuel distribution - such as lfw_fuel.

## Issues

This project is just getting started, so the API is subject to change, documentation is lacking, and options are not necessarily discoverable. I'm not so pleased with the hdf5 file sizes. The dev fuel dependency isn't great,
but this cannot be fixed until a fuel release. The overall software architecture is also rough, but it functions
fine a s proof of concept that can be refined if useful.

## Feedback:

Kerosene is currently an experiment in making datasets large and small easily sharable. Feedback welcome via github issues or email.
