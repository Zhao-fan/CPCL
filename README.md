# “[Complementary Parts Contrastive Learning for Fine-grained Weakly Supervised Object Co-localization](https://ieeexplore.ieee.org/document/10098208)” has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT, 2023).
## Abstract
The aim of weakly supervised object co-localization is to locate different objects of the same superclass in a dataset. Recent methods achieve impressive co-localization performance by multiple instance learning and self-supervised learning. However, these methods ignore the common part information shared by fine-grained objects and the influence of the complementary parts on the co-localization of the fine-grained objects. To solve these issues, we propose a complementary parts contrastive learning method for fine-grained weakly supervised object co-localization. The proposed method follows such an assumption that fine-grained object parts with the same/different semantic meaning should have similar/dissimilar feature representations in the feature space. The proposed method tackles two critical issues in this task: i) how to spread the model's attention and suppress the complex background noise, and ii) how to leverage the cross-category common parts information to mitigate the context co-occurrence problem. To address i), we attempt to integrate local and context cues via three types of attention including self-supervised attention, channel, and spatial attention to spread the model’s attention toward automatically identifying and localizing most discriminative parts of objects in the fine-grained images. To solve ii), we propose a cross-category object complementarity part contrastive learning module to identify the extracted part regions with different semantic information by pulling the same part features closer and pushing different part features away, which can mitigate the confounding bias caused by the co-occurrence surroundings within specific classes. Extensive qualitative and quantitative evaluations demonstrate the effectiveness of the proposed method on four fine-grained co-localization datasets: CUB-200-2011, Stanford Cars, FGVC-Aircraft, and Stanford Dogs.


## Requirements
* [Python 3.8](https://www.python.org/) <br>
* [PyTorch 1.8.0](https://pytorch.org/) <br>
* [CUDA 11.1](https://developer.nvidia.com/cuda-downloads) <br>
* [OpenCV 4.5](https://opencv.org/) <br>
* [Numpy 1.21](https://numpy.org/) <br>
* [Albumentations 0.5](https://github.com/albumentations-team/albumentations) <br>
* [Apex](https://github.com/NVIDIA/apex)

## Datasets
* Download the following datasets 
  * [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) <br>
  * [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) <br>
  * [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) <br>
  * [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) <br>
