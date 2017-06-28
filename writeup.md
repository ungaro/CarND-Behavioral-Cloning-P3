** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn.jpg "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[rec1]: ./examples/center_2017_06_28_00_27_28_030.jpg "Recovery Image"
[rec2]: ./examples/center_2017_06_28_00_27_28_736.jpg "Recovery Image"
[rec3]: ./examples/center_2017_06_28_00_27_29_228.jpg "Recovery Image"

[center1]: ./examples/center_2017_06_28_00_19_06_916.jpg "Center 1"
[center2]: ./examples/center_2017_06_28_00_19_06_995.jpg "Center 2"
[center3]: ./examples/center_2017_06_28_00_19_07_072.jpg "Center 3"
[center4]: ./examples/center_2017_06_28_00_19_07_150.jpg "Center 4"

[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've used NVIDIA's CNN architecture.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers for each CONV2D and fully-connected layers (except for the last one) in order to reduce overfitting (model.py lines 251-266). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 187-228, gen_val_data function). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and backwards driving. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I wanted to use a well established model for this assignment. NVIDIA's CNN was mentioned in the class so i choose it to see if i can implement and understand it.

At first my car would only go straight and not take turns. I thought this was because i've needed some recovery turns data, but it turned out that my model was underfitting. I've raised number of epochs, samples_per_epoch and batch size accordingly and used Keras EarlyStopping callback to not lose time if there was no reduction on validation loss.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

|  Layer (type)  			|  Output Shape  		|	 Param #		|
|---------------------------|-----------------------|-------------------|
|Normalization (Lambda)     |  (None, 66, 200, 3)   |     0				|
|conv2d_1 (Conv2D)          |  (None, 31, 98, 24)   |     1824			|
|dropout_1 (Dropout)        |  (None, 31, 98, 24)   |     0				|
|conv2d_2 (Conv2D)          |  (None, 14, 47, 36)   |     21636			|
|dropout_2 (Dropout)        |  (None, 14, 47, 36)   |     0				|
|conv2d_3 (Conv2D)          |  (None, 5, 22, 48)    |     43248			|
|dropout_3 (Dropout)        |  (None, 5, 22, 48)    |     0				|
|conv2d_4 (Conv2D)          |  (None, 3, 20, 64)    |     27712			|
|dropout_4 (Dropout)        |  (None, 3, 20, 64)    |     0				|
|conv2d_5 (Conv2D)          |  (None, 1, 18, 64)    |     36928			|
|dropout_5 (Dropout)        |  (None, 1, 18, 64)    |     0				|
|flatten_1 (Flatten)        |  (None, 1152)         |     0				|
|dense_1 (Dense)            |  (None, 100)          |     115300		|
|dropout_6 (Dropout)        |  (None, 100)          |     0				|
|dense_2 (Dense)            |  (None, 50)           |     5050			|
|dropout_7 (Dropout)        |  (None, 50)           |     0				|
|dense_3 (Dense)            |  (None, 10)           |     510			|
|dropout_8 (Dropout)        |  (None, 10)           |     0				|
|Out (Dense)                |  (None, 1)            |     11			|
-------------------------------------------------------------------------

- Total params: 252,219
- Trainable params: 252,219
- Non-trainable params: 0


Here is a visualization of the architecture from NVIDIA's own paper. http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf Page-5

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 7-8 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center1]
![alt text][center2]
![alt text][center3]
![alt text][center4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center in turns so that the vehicle would learn to recover. These images show what a recovery looks like starting from a curve.

![alt text][rec1]
![alt text][rec2]
![alt text][rec3]

I shifted the camera images horizontally to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle. I added 0.004 steering angle units per pixel shift to the right, and subtracted 0.004 steering angle units per pixel shift to the left. I will also shift the images vertically by a random number to simulate the effect of driving up or down the slope.

Ref:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

After the collection process, I had 48.045 number of data points. I then preprocessed this data by cropping the images to fit NVIDIA architecture.

I finally randomly shuffled the data set and put data from the second track into the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I've used Keras EarlyStopping to get callback to Stop training when a monitored quantity has stopped improving. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Lastly, I've not used video.py nor model.py to generate images because that is creating a bottleneck on the computer and drive.py can not respond timely to driving data resulting wobbly behaviour of the car. Instead i've used OBS Studio to capture the Udacity Self Driving Car Sim. window.

On github there is a small 1 minute version of the video, and i've made a longer version available on youtube.
I've also added a background music for the reviewer since it's incredibly boring to watch a car drive by itself in the simulator. :)

https://www.youtube.com/watch?v=1iLcs5K1keQ

The original music video is here:
https://www.youtube.com/watch?v=KoUEICCgwwA