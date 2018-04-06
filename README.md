# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_dir/data_vis.png "Visualization"
[image2]: ./writeup_dir/mean_image.png "Mean Image"
[image3]: ./writeup_dir/stddev_image.png "Stddev Image"
[image4]: ./writeup_dir/orig_image.png "Original Image"
[image5]: ./writeup_dir/rot_image1.png "Rotated Image1"
[image6]: ./writeup_dir/rot_image2.png "Rotated Image2"
[image7]: ./web_examples/240px-Zeichen_274-60_-_Zulässige_Höchstgeschwindigkeit,_StVO_2017.jpeg
[image8]: ./web_examples/240px-Zeichen_276_-_Überholverbot_für_Kraftfahrzeuge_aller_Art,_StVO_1992.jpeg
[image9]: ./web_examples/240px-Zeichen_301_-_Vorfahrt,_StVO_1970.svg.jpeg
[image10]: ./web_examples/Zeichen_103-10_-_Kurve_(links),_StVO_1992.svg.jpeg
[image11]: ./web_examples/Zeichen_267_-_Verbot_der_Einfahrt,_StVO_1970.jpeg
[image12]: ./writeup_dir/softmax1.png
[image13]: ./writeup_dir/softmax2.png
[image14]: ./writeup_dir/softmax3.png
[image15]: ./writeup_dir/softmax4.png
[image16]: ./writeup_dir/softmax5.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and the project code is submitted online as html and ipynb.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410 
* The size of test set is 12630 
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43 

#### 2. Include an exploratory visualization of the dataset.

I printed out the counts of each of 43 signs. A sample is shown below. Please see notebook for full list. 

```
Traffic Sign Id, Traffic Sign Label, Count in Training Set
0, Speed limit (20km/h), 180
1, Speed limit (30km/h), 1980
2, Speed limit (50km/h), 2010
3, Speed limit (60km/h), 1260
4, Speed limit (70km/h), 1770
5, Speed limit (80km/h), 1650
```

Here is an exploratory visualization of the data set. It is a bar chart showing how the counts of most common 5 signs 

![alt text][image1]

I also showed 5 randomly selected images and calculated their per-channel means. I also calculated and showed mean and stddev images of the training data set.

![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, the training data was shuffled. Next, if not already done, training data were augmented. For each image in training data, two new images were added one rotated by +15 degrees and another rotated by -15 degrees. Next, if not already done, following operations were applied to training, validation and the test set :

* scaling pixel value from 0-255 to 0.1 - 0.9. 
* removing per channel brightness from each image. This is done by calculating per channel mean and substracting it from each channel of the image. This is so that learning is not sensitive to brightness rather to the contrast in each image
* normalize all the images by substracting the mean of the images and diving by standard deviation of all images. This is so that all features have zero mean and unit variance. This is so that weights can be shared in each layer and no one feature heavily influence the weights. 

I chose not to convert the images to grayscale because colors are relevant for traffic signs, for example a large amount of red is likely to be a stop sign.

I decided to generate additional data because adding more data would likely help train the network better. Here is an example of an original image and rotated images:

![alt text][image4]
![alt text][image5]
![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer Filter          |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Color Map 1x1         | 1x1 stride, same padding, outputs 32x32x3     |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Flatten1              | Flattens output of 1st pooling layer          |
| Flatten2              | Flattens output of 2nd pooling layer          |
| Concatenate           | Concatenates Flatten1 and Flatten2            |
| Fully connected		| Input 1576, Output 1000                       |
| Fully connected       | Input 1000, Output 400                        |
| Fully connected       | Input 400, Output 43                          |
| Softmax				| Outputs softmax probabilities        			|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used number of epochs = 50, batch size = 128, learning rate is 0.001 with an AdamOptimizer. The loss function is cross entropy of softmax probabilities and one hot labels.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999 
* validation set accuracy of 0.949 
* test set accuracy of 0.929

The first architecture chosen was just LeNet architecture from LeNet Lab, changing the input from 32x32x1 to 32x32x3 and changing the outputs from 10 to 43. Using that I was able to about 0.9 accuracy on validation set. Based on upon reading up discussions and some pointers from my mentor, I first experimented with preprocessing steps, such as changing the scaling range from 0-255 to [0,1] or [-0.5,0.5]. The range 0.1 to 0.9 seem to work best. I also introduced other preprocessing steps such as substrating per channel mean from each image, removing relative brightness from each image. I also normalized the images by substracting the mean of all images and dividing by standard deviation of images. This preprocessing step was applied to training, validation and test set. This improved the validation accuracy but I was still not able to get to 0.93. Next I augmented the training set by adding two rotated images for every image in training set.

I also changed the architecture of the network in the following ways:

* Added a 1x1x3x3 (width x height x input channels x output) convolution layer that learns a color map for this application
* Forwarded the output of first maxpooling layer to the later layers along with the output of the second maxpooling layer. This would help the model learn from both high level features as well as low level features.
* Changed the dimensions of fully connected layers as well added another fully connected layer. The validation accuracy seem to be better when another fully connected layer is added.

Some of these changes were based on insights provided in this article, pointed to me by my mentor : https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image9] ![alt text][image11] 
![alt text][image8] ![alt text][image10]

These images do not seem to be part of the training set. Unlike the training set data, these images do not seem to be taken from real roads. Rather, they seem to be "golden" traffic signs as distributed in diriving education material. These images do not have real life noises as training set images have. Therefore, a network that learns from real life traffic sign images taken from roads may have difficulty predicting such "golden" traffic signs. 
 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h	      		| 60 km/h				 				        |
| No Passing      		| No Passing  									| 
| No Entry     			| No Entry 										|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Dangerous curve to the left | Dangerous curve to the left	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9% 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16-18 cells of the Ipython notebook.

For the first image, model is quite sure that this is 60km/hr speed limit sign, as its probability is 0.997, while next 4 maximum probabilities are 0.001 or less. The same trend is seen for 3rd, 4th and 5th images. However, for the 2nd image, the model is relatively less sure of its prediction, even though its prediction is correct, maximum probability being 0.9, while second highest probability is 0.09. The top 5 softmax probabilities for predictions of each image is shown below, in the same order as above. 

![alt text][image7]![alt text][image12]
![alt text][image8]![alt text][image13]
![alt text][image9]![alt text][image14]
![alt text][image10]![alt text][image15]
![alt text][image11]![alt text][image16]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


