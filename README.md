# PoE-based variational autoencoders
The source code for "Multimodal Variational Autoencoders for Semi-Supervised Learning: In Defense of Product-of-Experts"

The `misc` folder contains precomputed vectorized representations for  CUB-captions dataset and pretrained oracle networks for MNIST and SVHN. 

The data for CUB-Captions can be downloaded from <http://www.robots.ox.ac.uk/~yshi/mmdgm/datasets/cub.zip>

There are four PoE models available for training:

- **VAEVAE** from Wu et al. "Multimodal generative models for compositional representation
learning"
- **SVAE** from the current paper
- **VAEVAE_star** - VAEVAE architecture and SVAE loss function
- **SVAE_star** - SVAE architecture and VAEVAE loss function

The command line template for training the model is:

```bash
python experiments/<experiment_name>/run.py <model_name> <share of unpaired samples> <optional: evaluation mode>
```

For example

```
python experiments/mnist_split/run.py SVAE 0.9
```
will run the training for SVAE model with 10% supervision level


```
python experiments/mnist_split/run.py SVAE 0.9 eval best
```
will generate evaluation metrics and images for the best epoch of the training

```
python experiments/mnist_split/run.py SVAE 0.9 eval current
```
will generate evaluation metrics and images for the last epoch