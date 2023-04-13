# “[Complementary Parts Contrastive Learning for Fine-grained Weakly Supervised Object Co-localization](https://ieeexplore.ieee.org/document/10098208)” has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT, 2023).
## The Framework
![](https://github.com/Zhao-fan/CPCL/blob/main/images/framework.png)
The framework of CPCL. Please refer to [Paper Link](https://ieeexplore.ieee.org/document/10098208) for details.

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

## Training
Training the model as below:
```
cd classifier     
python train_c.py         # train the classification model
python test_c.py          # keep the classification result to top1top5.npy  

cd ../gengration         
python train_g.py         # train the Pseudo-label Generation Network  
python test_g.py          # keep the pseudo masks 

cd ../localization        
python train_l.py         # train the class-agnostic co-localization Network 
python test_l.py          # evaluate the localization accuracy 
```
* If you want to train your own model, please download the [pretrained model](https://download.pytorch.org/models/resnet50-19c8e357.pth) into `resource` folder.

## Testing
Testing the trained model as below:
```
cd localization
python test_l.py         # evaluate the model performance 
```
* If you want to evaluate the performance of CPCL, please download our trained model:
  * [CUB-200-2011](https://pan.baidu.com/s/1tGZqun_oEth-UbwM2zIFlw) extract code：h747, put it into the folder `localization/out`   <br>
  * [Stanford Cars](https://pan.baidu.com/s/19I1314zLyBPJQkHmXvvNgA) extract code：a889, put it into the folder `localization/out`  <br>
  * [FGVC-Aircraft](https://pan.baidu.com/s/1uujnylWIdwBO2uyY1yTPmQ) extract code：a79f, put it into the folder `localization/out`  <br>
  * [Stanford Dogs](https://pan.baidu.com/s/1o_3D3xfJW5kfC-I441KcMg) extract code：v2as, put it into the folder `localization/out`  <br> 

## Quantitative comparisons
![](https://github.com/Zhao-fan/CPCL/blob/main/images/cubloc.png)

## Qualitative comparisons
![](https://github.com/Zhao-fan/CPCL/blob/main/images/Fig_6.png)
