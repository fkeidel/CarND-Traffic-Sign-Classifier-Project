# **Traffic Sign Recognition** 

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[random_train]: ./writeup-images/writeup_20_random_train_samples.png "20 random train samples"
[samples_distribution]: ./writeup-images/writeup_samples_distribution.png "train samples distribution"
[grayscale]: ./writeup-images/writeup_grayscale.png "grayscale"
[equalized]: ./writeup-images/writeup_equalize.png "equalized"
[training_process]: ./writeup-images/writeup_training_process.png "training process"
[new_samples]: ./writeup-images/writeup_new_samples.png "new samples"
[new_samples_preprocessed]: ./writeup-images/writeup_new_samples_preprocesses.png "new samples preprocessed"
[visualization_input]: ./writeup-images/writeup_visualization_input.png "visualization input"
[visualization_layer1]: ./writeup-images/writeup_visualization_layer1.png "visualization layer 1"
[visualization_layer2]: ./writeup-images/writeup_visualization_layer2.png "visualization layer2"

## Rubric Points

### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one. 

This is the writeup for the project *Traffic Sign Classifier*. Here you can find the [project code](https://github.com/fkeidel/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
For a first impression, I printed out 20 random samples of the training set:

![alt text][random_train]

As can be seen from above, the training samples contain many images with bad contrast (i.e. dark or 'foggy').

To see, how many samples from each class exist in the training set, I printed out a histogram.

![alt text][samples_distribution]

The histogram shows the samples distribution. As can be seen from the diagram, the samples of each class have not the same sample count. This could lead to a bias to recognizing better the classes with higher sample count.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale, because in the paper [Trafﬁc Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) the authors mentioned, that they had better results of their CNN model with grayscale images. Although, using grayscale showed no significant effect during training, I kept the grayscaling step because it reduces the number of weights in the first network layer.

![alt text][grayscale]

As mentioned above, many images of the training set have bad contrast. So, as a second preprocessing step, I improved contrast and brightness by equalizing the gray values in the images with the function cv2.equalizeHist(). As a human, it is easier to recognize images with a good contrast. So I suppose, it will be easier for a Neural Network, too.

![alt text][equalized]

As a last step, I normalized the gray values of all images to lie in a symmetric range [-1,1]. 

The preprocessing steps increased my model performance by 3%.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, output 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, output 5x5x16 				|
| flatten | output 400 |
| Fully connected		1 | input = 400, output = 120	|
| RELU					|												|
| Dropout					| keep prob = 0.5 during training			|
| Fully connected		2 | Input = 120, output = 84	|
| RELU					|												|
| Dropout					| keep prob = 0.5 during training			|
| Fully connected		3 | Input = 84, output = 43	|
| Softmax				|         									|
 
#### 3. Describe how you trained your model. 

To train the model, I calculated the cross-entropy of the calculated logits and the one-hot-encoded class labels. As a loss function, I used the mean of the entropies of all classes. As optimizer, I used the AdamOptimizer. 

The starting values for the hyperparameters were:
Epochs = 10
Batch size = 128.
Learning rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

As a starting point, I used the LeNet convolutional network architecture, consisting of 2 convolutional layers and 3 fully connected layers. The LeNet architecture, with the starting values of the hyperparameters, had a performance of 87%.

Then, for each hyperparameter, I increased the parameter a little bit and measured the validation performance. I the result was better than the previous one, I increase the parameter even more until it reached a maximum. If increasing the hyperparamter didn't increase the result, I went in the opposite direction and decreased the parameter starting from the initial value until the maximum found.

Lowering the batch size from 128 to 64, increased the model performance by 3% to 90%.

Preprocessing the images (grayscale and normalization to a symmetric range of [-1,1]) further increased the performance by 2% to 92%.

As additional improvement, I used 50% dropout in the fully connected layers during training. This increased the model performance by 3% to 95%.

My final model results were:
* training set accuracy of 95%
* validation set accuracy of 95% 
* test set accuracy of 93.4%

The following figure shows the training process with the final architecture and hyperparameters:

![alt text][training_process]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. 

![alt text][new_samples]

I used the same preprocessing steps as when training the network.

![alt text][new_samples_preprocessed]

The first image might be difficult to classify because the road sign's orientation is not perpendicular to the optical axis. The other signs should have no problems to be recognized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is better than the accuracy on the test set.

Detected road signs:
1. No entry
2. Right-of-way at the next intersection
3. Speed limit (60km/h)
4. End of all speed and passing limits
5. Yield

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.6%         			| No entry   									| 
| 94.5%     				| Right-of-way at the next intersection										|
| 77.0%					| Speed limit (60km/h)										|
| 90.6%      			| End of all speed and passing limits				 				|
| 100%				    | Yield    							|

Suprisingly, the first sign has a high probability for the correct class (99.6%) even if the orientation of the sign was not optimal. A very high probability of nearly 100% showed the Yield sign.

### (Optional) Visualizing the Neural Network 
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

For visualizing the feature maps of the convolution layers, I selected the second image of the new images (Right-of-way at the next intersection)

![alt text][visualization_input]

The feature map of the first layer looked like this:

![alt text][visualization_layer1]

And the feature map of the second layer showed up like this:

![alt text][visualization_layer2]

The feature map of the first layer shows significant shapes in the input image (lines). The feature map of the second layer is difficult to interpret.
