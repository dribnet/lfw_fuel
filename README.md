# LFW dataset, converted to fuel

[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) is a database of face photographs
designed for studying the problem of unconstrained face recognition.

This project currently packages the pairsDevTrain / pairsDevTest
image sets into a fuel compatible dataset along with targets
to indicate whether the pairs are same or different. In addition
to the original lfw dataset, conversion is supported for both
the funneled and deepfunneled versions of the images.

This project uses [kerosene](https://github.com/dribnet/kerosene) to produce a [fuel](https://github.com/mila-udem/fuel) comptable
hdf5 file that is usable by [blocks](https://github.com/mila-udem/blocks) or [keras](https://github.com/fchollet/keras).

## Show me

From the [included example](https://github.com/dribnet/lfw_fuel/blob/master/example/run-lfw.py)

```python
from keras.models import Sequential
from lfw_fuel import lfw

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = lfw.load_data(format="deepfunneled")

# (build the perfect model here)

model.fit(X_train, Y_train, show_accuracy=True, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
```

The features are currently stored in six channels - three for each
of the two RGB images to be compared.

Note that the images are `250x250` - which is quite large by most
CNN standards. These can be cropped and scaled before passing them
to the network as shown in the example.

## What's this dataset all about again?

The primary task of Labeled Faces in the Wild is to learn wheather the face
in two pictures are the same person, or two different people. There are
2200 training pairs and 1000 test pairs in the predefined split.

Here are three matching training pairs:

| Image 1 | Image 2 | Status |
|---------|---------|--------|
| ![Aaron_Peirsol_0003](http://vis-www.cs.umass.edu/lfw/images/Aaron_Peirsol/Aaron_Peirsol_0003.jpg "Aaron_Peirsol_0003") | ![Aaron_Peirsol_0004](http://vis-www.cs.umass.edu/lfw/images/Aaron_Peirsol/Aaron_Peirsol_0004.jpg "Aaron_Peirsol_0004") | MATCH |
| ![Aaron_Sorkin_0001](http://vis-www.cs.umass.edu/lfw/images/Aaron_Sorkin/Aaron_Sorkin_0001.jpg "Aaron_Sorkin_0001") | ![Aaron_Sorkin_0002](http://vis-www.cs.umass.edu/lfw/images/Aaron_Sorkin/Aaron_Sorkin_0002.jpg "Aaron_Sorkin_0002") | MATCH |
| ![Abdel_Nasser_Assidi_0001](http://vis-www.cs.umass.edu/lfw/images/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0001.jpg "Abdel_Nasser_Assidi_0001") | ![Abdel_Nasser_Assidi_0002](http://vis-www.cs.umass.edu/lfw/images/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.jpg "Abdel_Nasser_Assidi_0002") | MATCH |

And here are three non-matching training pairs

| Image 1 | Image 2 | Status |
|---------|---------|--------|
| ![Lee_Nam-shin_0001](http://vis-www.cs.umass.edu/lfw/images/Lee_Nam-shin/Lee_Nam-shin_0001.jpg "Lee_Nam-shin_0001") | ![Nick_Nolte_0001](http://vis-www.cs.umass.edu/lfw/images/Nick_Nolte/Nick_Nolte_0001.jpg "Nick_Nolte_0001") | DIFFERENT |
| ![Lee_Soo-hyuck_0001](http://vis-www.cs.umass.edu/lfw/images/Lee_Soo-hyuck/Lee_Soo-hyuck_0001.jpg "Lee_Soo-hyuck_0001") | ![Scott_Sullivan_0001](http://vis-www.cs.umass.edu/lfw/images/Scott_Sullivan/Scott_Sullivan_0001.jpg "Scott_Sullivan_0001") | DIFFERENT |
| ![Lee_Yeo-jin_0001](http://vis-www.cs.umass.edu/lfw/images/Lee_Yeo-jin/Lee_Yeo-jin_0001.jpg "Lee_Yeo-jin_0001") | ![Mariangel_Ruiz_Torrealba_0001](http://vis-www.cs.umass.edu/lfw/images/Mariangel_Ruiz_Torrealba/Mariangel_Ruiz_Torrealba_0001.jpg "Mariangel_Ruiz_Torrealba_0001") | DIFFERENT |


In addition, this dataset is provided in both this raw format, and at
least two "preprocessed" versions called `funneled` and `deepfunneled`.
Often these are very similar, but here is an example of how they can differ.

| Original | Funneled | Deep Funneled |
|---------|---------|--------|
| ![Amelia_Vega_0004](http://vis-www.cs.umass.edu/lfw/images/Amelia_Vega/Amelia_Vega_0004.jpg "Amelia_Vega_0004") | ![Amelia_Vega_0004 Funneled](http://vis-www.cs.umass.edu/lfw/images_funneled/Amelia_Vega/Amelia_Vega_0004.jpg "Amelia_Vega_0004 Funneled") | <img alt="Amelia_Vega_0004 Deep Funneled" src="http://vis-www.cs.umass.edu/lfw/images_deepfunneled/Amelia_Vega/Amelia_Vega_0004.jpg" width="150" height="150" /> |

On the LFW page you can browse the [complete training set](http://vis-www.cs.umass.edu/lfw/devTrain.html) or the [complete test set](http://vis-www.cs.umass.edu/lfw/devTest.html) and see all three versions of all images.

## Example

There is an included example of how to train a network using
keras for this task. To run this example from the repo:

```bash
$ python example/run-lfw.py
```

This should run the example, downloading the dataset if necessary.

Note that currently the example runs, but the performance is poor.
Suggestions or merge requests improving this example certainly welcome.

## Installation

Installation is optional - if kerosene is installed then simply clone
the repo and run the example script. However, installation is an option
so that the lfw_fuel dependency can be used from the path, which can
be useful if you'd like to use this dataset in your own blocks or
keras project.

```
python setup.py install
```

You can also rebuild the hdf5 files from scratch by running the
`kero-download` and `kero-convert` scripts, which are currently
part of the kerosene installation. For example:

```bash
kero-download lfw_fuel/lfw.py
kero-convert lfw_fuel/lfw.py
```

This will convert the original version of lfw, but funneled and
deepfunneled formats are also supported:

```bash
kero-download lfw_fuel/lfw.py --format deepfunneled
kero-convert lfw_fuel/lfw.py --format deepfunneled
```

## License

MIT
