
# Photo Calculator

## Introduction
This project uses a convolutional neural network (CNN) to recognize math symbols in handwritten images. The dataset used for training and testing the model is a collection of images of handwritten math symbols. The model is implemented using Tensorflow and Keras.

## Data Preparation and Preprocessing
The first step in this project is to prepare the data for training and testing the model. The images are read in using the OpenCV library and are labeled based on their corresponding math symbol. A visual representation of the data is also provided to give an idea of the distribution of the different math symbols in the dataset.

The images are then processed using Otsu's thresholding method to make the symbols in the images more distinct and easier for the model to recognize. The labels are also encoded to prepare them for training.

## Model Architecture
The model architecture used in this project is a simple CNN architecture with multiple convolutional and pooling layers followed by a fully connected layer.

## Model Training and Evaluation
The model is trained using a combination of K-fold cross validation and data augmentation techniques. The model's performance is evaluated using classification report and confusion matrix metrics.

### How to Run
- Download the project files 
 - Unzip the data_berke.zip 
- Run the project on your machine
#### Dependencies
- opencv-python
 - imutils 
 - tensorflow 
- seaborn
 - matplotlib
- pandas
- numpy
- Pillow
- sklearn
- IPython

#### Note
- The model is trained on a specific dataset, and it may not perform well on other datasets.
- The model is trained on a specific set of symbols, it may not be able to recognize other symbols.
 - The code uses a pre-trained model, if you want to train it again you should delete the weight files and re-run the code.
- This is a simple model and it may not perform well on real-world data.
