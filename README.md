# CondenseNets

This repository contains the code (in PyTorch) for "[CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/abs/1711.09224)" paper by [Gao Huang](http://www.cs.cornell.edu/~gaohuang/)\*, [Shichen Liu](https://shichenliu.github.io)\*, [Laurens van der Maaten](https://lvdmaaten.github.io) and [Kilian Weinberger](https://www.cs.cornell.edu/%7Ekilian/) (* Authors contributed equally).

### Citation

If you find our project useful in your research, please consider citing:

```
@inproceedings{huang2018condensenet,
  title={Condensenet: An efficient densenet using learned group convolutions},
  author={Huang, Gao and Liu, Shichen and Van der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2752--2761},
  year={2018}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Discussions](#discussions)
5. [Contacts](#contacts)

## Introduction

CondenseNet is a novel, computationally efficient convolutional network architecture. It combines dense connectivity between layers with a mechanism to remove unused connections. The dense connectivity facilitates feature re-use in the network, whereas learned group convolutions remove connections between layers for which this feature re-use is superfluous. At test time, our model can be implemented using standard grouped convolutions —- allowing for efficient computation in practice. Our experiments demonstrate that CondenseNets are much more efficient than other compact convolutional networks such as MobileNets and ShuffleNets.

<img src="https://user-images.githubusercontent.com/9162722/32978657-b10fae0e-cc81-11e7-888d-1f9e4c028a9b.png">

Figure 1: Learned Group Convolution with G=C=3.

<img src="https://user-images.githubusercontent.com/9162722/31302319-6ca3a49c-ab33-11e7-938c-70379feca5bc.jpg" width="480">

Figure 2: CondenseNets with Fully Dense Connectivity and Increasing Growth Rate.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(1.1.0)](http://pytorch.org)
- [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/)

### Train
As an example, use the following command to train a CondenseNet on ImageNet

```
python main.py --model condensenet -b 256 -j 20 /PATH/TO/IMAGENET \
--stages 4-6-8-10-8 --growth 8-16-32-64-128 --gpu 0,1,2,3,4,5,6,7 --resume
```

As another example, use the following command to train a CondenseNet on CIFAR-10

```
python main.py --model condensenet -b 64 -j 12 cifar10 \
--stages 14-14-14 --growth 8-16-32 --gpu 0 --resume
```


### Evaluation
We take the ImageNet model trained above as an example.

To evaluate the trained model, use `evaluate` to evaluate from the default checkpoint directory:

```
python main.py --model condensenet -b 64 -j 20 /PATH/TO/IMAGENET \
--stages 4-6-8-10-8 --growth 8-16-32-64-128 --gpu 0 --resume \
--evaluate
```

or use `evaluate-from` to evaluate from an arbitrary path:

```
python main.py --model condensenet -b 64 -j 20 /PATH/TO/IMAGENET \
--stages 4-6-8-10-8 --growth 8-16-32-64-128 --gpu 0 --resume \
--evaluate-from /PATH/TO/BEST/MODEL
```

Note that these models are still the large models. To convert the model to group-convolution version as described in the paper, use the `convert-from` function:

```
python main.py --model condensenet -b 64 -j 20 /PATH/TO/IMAGENET \
--stages 4-6-8-10-8 --growth 8-16-32-64-128 --gpu 0 --resume \
--convert-from /PATH/TO/BEST/MODEL
```

Finally, to directly load from a converted model (that is, a CondenseNet), use a **converted model file** in combination with the `evaluate-from` option:

```
python main.py --model condensenet_converted -b 64 -j 20 /PATH/TO/IMAGENET \
--stages 4-6-8-10-8 --growth 8-16-32-64-128 --gpu 0 --resume \
--evaluate-from /PATH/TO/CONVERTED/MODEL
```

### Other Options
We also include DenseNet implementation in this repository.  
For more examples of usage, please refer to [script.sh](script.sh)  
For detailed options, please `python main.py --help`

## Results

### Results on ImageNet

| Model | FLOPs | Params | Top-1 Err. | Top-5 Err. | Pytorch Model |
|---|---|---|---|---|---|
| CondenseNet-74 (C=G=4) | 529M | 4.8M | 26.2 | 8.3 | [Download (18.69M)](https://www.dropbox.com/s/sj26rm4so3uhdmg/converted_condensenet_4.pth.tar?dl=0) |
| CondenseNet-74 (C=G=8) | 274M | 2.9M | 29.0 | 10.0 | [Download (11.68M)](https://www.dropbox.com/s/aj1xpd6zcnclous/converted_condensenet_8.pth.tar?dl=0) |

### Results on CIFAR

| Model | FLOPs | Params | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| CondenseNet-50 | 28.6M | 0.22M | 6.22 | - |
| CondenseNet-74 | 51.9M | 0.41M | 5.28 | - |
| CondenseNet-86 | 65.8M | 0.52M | 5.06 | 23.64 |
| CondenseNet-98 | 81.3M | 0.65M | 4.83 | - |
| CondenseNet-110 | 98.2M | 0.79M | 4.63 | - |
| CondenseNet-122 | 116.7M | 0.95M | 4.48 | - |
| CondenseNet-182* | 513M | 4.2M | 3.76 | 18.47 |

(* trained 600 epochs)

### Inference time on ARM platform

| Model | FLOPs | Top-1 | Time(s) |
|---|---|---|---|
| VGG-16 | 15,300M | 28.5 | 354 |
| ResNet-18 | 1,818M | 30.2 | 8.14 |
| 1.0 MobileNet-224 | 569M | 29.4 | 1.96 |
| CondenseNet-74 (C=G=4) | 529M | 26.2 | 1.89 |
| CondenseNet-74 (C=G=8) | 274M | 29.0 | 0.99 |

## Contact
liushichen95@gmail.com  
gh349@cornell.com

We are working on the implementation on other frameworks.  
Any discussions or concerns are welcomed!
