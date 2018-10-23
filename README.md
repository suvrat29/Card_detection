# How To Train an Object Detection Classifier for Card-like Objects Using TensorFlow (GPU) on Ubuntu Linux(18.04 LTS)

## Brief Summary

This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifer for Card-like objects with bounding boxes on Ubuntu Linux(18.04 LTS). It is written in the latest TensorFlow version available on conda repository(1.10), but will also work for newer/older(upto 1.5) versions of TensorFlow. 

*Note: The below mentioned steps can also be undertaken in Windows 10, but currently there are some issues pertaining to the compatibility of the TensorFlow prebuilt binaries used in this project.*

This readme describes every step required to get going with your own object detection classifier.

**Table of contents:**
1. [Installing TensorFlow-GPU]()
2. [Setting up the Object Detection directory structure and Anaconda Virtual Environment]()
3. [Gathering and labeling pictures]()
4. [Generating training data]()
5. [Creating a label map and configure training]()
6. [Training]()
7. [Exporting the inference graph]()
8. [Testing and using the newly trained object detection classifier]()

This repository includes all the files needed to build a classifier based on the TensorFlow(GPU). Anything that causes any kind of issue has been rectified and the steps has been noted on the subsequent chapters with a "***Note:***". It also has the required python scripts that are used for tasks like converting xml to csv files or renaming the files.

## Introduction

The purpose of this readme is to explain and better understand how to train your own Convolutional Neural Network(CNN) object detection classifier for multiple objects, starting from zero. At the end of this readme, we would have built a program that can identify and draw boxes around specific objects in pictures.

This tutorial provides instruction for training a classifier that can detect multiple objects, not just one. The tutorial is written for Ubuntu 18.04 LTS, but it will also work on Ubuntu 16.04 and other distributions of linux. The general procedure can also be used for Windows operating systems, but file paths and package installation commands will need to be changed accordingly.

TensoFlow-GPU allows a PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. The general consensus points to the conclusion that using TensorFLow-GPU cuts down the training time factor by several times (3 hours instead of 24 hours) as opposed to using TesorFlow-CPU. Regular TensorFLow can also be used when following this guide, but the training time will be longer. If we use regular TensorFLow(TensorFLow-CPU), we don't need CUDA and cuDNN in step 1. I used TensorFLow-GPU v1.10 at the time of writing this guide, but it will likely work for the future versions of TensorFlow.

## Steps


### 1. Installing TensorFlow-GPU 1.10 (skip this step if TensorFlow-GPU already installed)

Install TensorFlow-GPU by floowing the instructions in [this very beautifully written tutorial by Pudget Systems.](https://www.pugetsystems.com/labs/hpc/Install-TensorFlow-with-GPU-Support-the-Easy-Way-on-Ubuntu-18-04-without-installing-CUDA-1170/).

Be sure to install Anaconda with Python 3.6 as instructed in the tutorial, as the Anaconda virtual environment will be used for the rest of this guide.

### 2. Set up TensorFlow Directory and Anaconda Virtual Environment

The TensorFlow Object Detection API has some restrictions regarding the directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or traing an object detection model.

This portion of the guide goes over the full setup required. It is fairly straightforward, but follow the instructions closely, because improper setup can cause very problematic and headache inducing errors down the road.

#### 2a. Download TensorFlow Object Detection API repository from GitHub

Create a folder in the '/home' directory and name it "TensorFlow". This working directory will containg the full TensorFlow object detection framework, as well as our training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

*Note: Make sure there are no spaces anywhere in the directory structure as the commands pertaining will fail due to incomplete path. Instead of using spaces when assigning directory names, use underscores(_).*

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the "Clone or Download" button and downloading the .zip file. This can also be done with the git cloning command from the terminal using:
```
git clone https://github.com/tensorflow/models.git
```
*Note: To use the "git clone" command, you should have the pre-requisite packages already installed in your ubuntu. This can be done by issuing the following commands:*
```
apt-get update
apt-get install git-core
```

#### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo

TensorFlow provides several object detection models(pre-trained classifiers with specific neural network architectures) in its model zoo. Some models have an architecture that allows for faster detection but with less accuracy, while some models give slower detection but with more accuracy. In this guide, we will be using the Faster-RCNN model, but one can choose a different model to train their object detection classifier. If we aim to run the object detector on a device with low computational power(like smartphones), we can use the SSD-MobileNet model. If we'll be running our detector on a decently powered laptop or desktop PC, we'll use one of the RCNN models.

[Download the model here](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz). Open the downloaded tar.gz file with an archiver such as 7-zip and extract the folder into /home/TensorFlow/models/research/object_detection folder.

**Compile using Protobuf**
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the TensorFlow/models/research/ directory:

```protoc object_detection/protos/*.proto --python_out=.```

#### 2c. Directory Structure

The Directory structure has two approaches, one is to place all the required files and scripts directly into the /TensorFlow/ directory, or we can create a separate directory on our /home/. In this guide we'll follow the latter way as it leaves the tensorflow directory untouched and the directory has a clean and separate structure without any mixing or matching of files & folders.

At this point the directory structure should be like:-
- /home/ProjectFloder/images/test - for the test images
- /home/ProjectFolder/images/train - for the train images
- /home/ProjectFolder/ - this directory will contain the scripts and files required to train the model
- /home/ProjectFolder/training - for the model configuration files and labelmaps

Now, we are ready to start training our very own object detector. We'll be moving on to understand how the various files will be generated and used for training the model from our own dataset.

#### 2d. Set up new Anaconda virtual environment

Working within a virtual environment is a very good practice as it leaves the installation of the python in our system untouched and all the modifications are done within that virtual environment. If we mess up at any stage, we can just delete the environment instead of reinstalling python in our system from scratch.

Open a terminal, and create a new environment called "tf-gpu" by issuing the following command:
```
conda create --name tf-gpu
```
Now, activate the environment using:
```
source activate tf-gpu
```
Install TensorFlow with GPU acceleration and all of the dependencies:
```
conda install tensorflow-gpu
```
**Create a new Jupyter Notebook Kernel for the TensorFlow environment.**
With your tf-gpu environment activated, issue:
```
conda install ipykernel
```
Now create the Jupyter Kernel:
```
python -m ipykernel install --user --name tf-gpu --display-name "TensorFlow-GPU"
```
With this "tf-gpu" kernel installed, when you open a Jupyter notebook you will now have an option to to start a new notebook with this kernel.

#### 2e. Configure PYTHONPATH environment variable

A PYTHONPATH variable must be created that points to /models, /models/research and /models/research/slim directories. To do this, the below mentioned steps have to be followed word to word, everytime right after activating the tf-gpu environment:
- Change the working directory to: /home/TensorFlow/models/research/
- Set PYTHONPATH variable as: ``` export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ```
- Type ``` cd``` to change to the regular working directory
- Start the jupyter notebook ``` jupyter notebook ```

#### 2f. Test Tensorflow setup to verify it works

The TensorFlow Object Detection API is now all set up to use pre-trained models for object detection, or to train a new one. You can test it out and verify your installation is working by launching the object_detection_tutorial.ipynb script with Jupyter. From the \object_detection directory, issue this command:
```
jupyter notebook object_detection_tutorial.ipynb
```

### 3. Gather and Label Pictures
