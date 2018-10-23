# How To Train an Object Detection Classifier for Card-like Objects Using TensorFlow (GPU) on Ubuntu Linux(18.04 LTS)

## Brief Summary
This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifer for Card-like objects with bounding boxes on Ubuntu Linux(18.04 LTS). It is written in the latest TensorFlow version available on conda repository(1.10), but will also work for newer/older(upto 1.5) versions of TensorFlow. 

*Note: The below mentioned steps can also be undertaken in Windows 10, but currently there are some issues pertaining to the compatibility of the TensorFlow prebuilt binaries used in this project.*

This readme describes every step required to get going with your own object detection classifier.

**Table of contents:**
1. [Installing TensorFlow-GPU](https://github.com/suvrat29/Card_detection#1-install-tensorflow-gpu-110-skip-this-step-if-tensorflow-gpu-already-installed)
2. [Setting up TensorFlow directory and Anaconda Virtual Environment]()
3. [Gathering and labeling pictures]()
4. [Generating training data]()
5. [Creating label map and configure training]()
6. [Run the Training]()
7. [Exporting the Inference Graph]()
8. [Testing and Using the Newly Trained Object Detection Classifier]()

This repository includes all the files needed to build a classifier based on the TensorFlow(GPU). Anything that causes any kind of issue has been rectified and the steps has been noted on the subsequent chapters with a "***Note:***". It also has the required python scripts that are used for tasks like converting xml to csv files or renaming the files.

## Introduction
The purpose of this readme is to explain and better understand how to train your own Convolutional Neural Network(CNN) object detection classifier for multiple objects, starting from zero. At the end of this readme, we would have built a program that can identify and draw boxes around specific objects in pictures.

This tutorial provides instruction for training a classifier that can detect multiple objects, not just one. The tutorial is written for Ubuntu 18.04 LTS, but it will also work on Ubuntu 16.04 and other distributions of linux. The general procedure can also be used for Windows operating systems, but file paths and package installation commands will need to be changed accordingly.

TensoFlow-GPU allows a PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. The general consensus points to the conclusion that using TensorFLow-GPU cuts down the training time factor by several times (3 hours instead of 24 hours) as opposed to using TesorFlow-CPU. Regular TensorFLow can also be used when following this guide, but the training time will be longer. If we use regular TensorFLow(TensorFLow-CPU), we don't need CUDA and cuDNN in step 1. I used TensorFLow-GPU v1.10 at the time of writing this guide, but it will likely work for the future versions of TensorFlow.

## Steps
### 1. Installing TensorFlow-GPU 1.10 (skip this step if TensorFlow-GPU already installed)
Install TensorFlow-GPU by following the instructions in [this very beautifully written tutorial by Pudget Systems.](https://www.pugetsystems.com/labs/hpc/Install-TensorFlow-with-GPU-Support-the-Easy-Way-on-Ubuntu-18-04-without-installing-CUDA-1170/).

Be sure to install Anaconda with Python 3.6 as instructed in the tutorial, as the Anaconda virtual environment will be used for the rest of this guide.

### 2. Setting up TensorFlow Directory and Anaconda Virtual Environment
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
- Paste all the .py files into the /home/ProjectFolder/ directory or else the python scripts will throw errors

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

### 3. Gathering and Labeling Pictures

Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.

#### 3a. Gather Pictures

TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the images along with the desired objects, and should have a variety of backgrounds and lightning conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture.

For my Card Detection classifier, I want to detect cards of different dimensions. I used my iPhone to take about 360 pictures in total with one, two or multiple cards in the frame, in varioud lightning conditions and partially obscured. To detect overlapping cards, I made sure to have the cards overlapped in many images.

You can use your phone to take pictures of the objects or download images of the objects from Google Image Search. I recommend having atleast 200 pictures overall.

*Note: Make sure the images aren't too large. They should be under or equal to a 720p resolution max, or else we run into serious RAM shortage in our training phase. For instance, I used the images straight from my iPhone without resizing them, and I ran into severe RAM shortage. Mind you, I used a 16GB config with 10GB swap area. The large sizes of images need ridiculous amount of RAM of the order of ~100GB+.*

To reduce the size and resolution, a script is available or you can use any service/software to resize your photos to 1280x720 resolution max. A website I used was https://bulkresizephotos.com/ as they claim to do the conversion client side, and is very quick.

#### 3b. Label Pictures

This part can be quick or very long depending on the number of images we have gathered. With all the pictures gathered, it is time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instruction on how to install and use it.

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Download and install LabelImg, point it to your /images/train directory, and then draw a box around each object in each image. Repeat the process for all the images in the /images/test directory.

LabelImg saves a .xml file which contains the label data for each image. The .xml files generated here are used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the /test and /train directories.

### 4. Generating Training Data

With the images labeled, it's time to generate the TFRecords that serve as input data to the TensorFlow training model. For this, we'll use xml_to_csv.py and generate_tfrecord.py scripts.

First, the image .xml data will be used to create .csv files containing all the labeling data for the train and test images. Place the /train and /test label .xml data in the structure /images/train/*.xml and /images/test/*.xml and then run the xml_to_csv.py script.
```
python xml_to_csv.py
```
 
 After running the script we get train_labels.csv and test_labels.csv in the /images/ folder.
 
 Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file.
 
 For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:
 ```
def class_text_to_int(row_label):
    if row_label == 'card':
        return 1
    else:
        return None
 ```
 With this:
 ```
def class_text_to_int(row_label):
    if row_label == 'scooter':
        return 1
    else:
        return None
 ```
 
 Then, generate the TFRecord files by issuing these commands from the /home/ProjectFolder/ folder:
 ```
 python generate_tfrecord.py --csv_input=/home/ProjectFolder/images/train_labels.csv --image_dir=/home/ProjectFolder/images/train --output_path=train.record
python generate_tfrecord.py --csv_input=/home/ProjectFolder/images/test_labels.csv --image_dir=/home/ProjectFolder/images/test --output_path=test.record
 ```
 
 These generate a train.record and a test.record file in /ProjectFolder. These will be used to train the new object detection classifier.
 
 ### 5. Creating Label Map and Configure Training
 
 The last thing to do before training is to create a label map and edit the training configuration file.
 
 #### 5a. Label Map
 
 The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the /home/ProjectFolder/training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Card Detector):
 ```
 item {
  id: 1
  name: 'card'
  }
 ```
 
 *Note: The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file.*
 
 #### 5b. Configure Training
 
 Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to /home/TensorFlow/models/research/object_detection/samples/configs and copy the faster_rcnn_inception_v2_pets.config file into the /home/ProjectFolder/training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the my model, it would be num_classes : 1.
- Line 110. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "/home/ProjectFolder/training/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
- Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "/home/ProjectFolder/train.record"
  - label_map_path: "/home/ProjectFolder/training/labelmap.pbtxt"
- Line 132. Change num_examples to the number of images you have in the **/images/test** directory.
- Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "/home/ProjectFolder/test.record"
  - label_map_path: "/home/ProjectFolder/training/labelmap.pbtxt"

Save the file after the changes have been made. The training job is all configured and ready to go.

### 6. Run the Training

*Note: As of version 1.9, TensorFlow has deprecated the "train.py" file and replaced it with "model_main.py" file. I haven't been able to get model_main.py to work correctly yet (I run in to errors related to pycocotools). Fortunately, the train.py file is still available in the /object_detection/legacy folder. Simply move train.py from /object_detection/legacy into the /object_detection folder and then continue following the steps below.*

To start training, issue the following command from /home/ProjectFolder/ directory to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins.

Each step of training reports the loss. It will start high and get lower and lower as training progresses. For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8. I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.

The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

### 7. Exporting the Inference Graph

Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the /home/ProjectFolder/ folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

This creates a frozen_inference_graph.pb file in the /home/ProjectFolder/inference_graph folder. The .pb file contains the object detection classifier.

### 8. Testing and Using the Newly Trained Object Detection Classifier

The object detection classifier is all ready to go! I’ve written Python scripts to test it out on an image, video, or webcam feed.

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. (For my Card Detector, there isone class I want to detect, so NUM_CLASSES = 1.)

To test your object detector, move a picture of the object or objects into the /ProjectFolder folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture.

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tf-gpu” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing any objects it’s detected in the image!


### I've included a jupyter notebook as well to better understand the code if the above instructions are, in some way, not clear.
### Credits to [EdjeElectronics](https://github.com/EdjeElectronics) as I've used his readme for pointers.
