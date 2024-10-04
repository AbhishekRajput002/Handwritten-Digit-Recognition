# Handwritten-Digit-Recognition 

**Overview**


This project focuses on recognizing handwritten digits using a machine learning model. The dataset is sourced from Kaggle and is designed to train a model that accurately predicts the digit shown in an image.

**Dataset**


The dataset used for this project can be downloaded from Kaggle:


**Dataset Name:**


     Kaggle Dataset Link: https://www.kaggle.com/competitions/digit-recognizer


The dataset contains images of handwritten digits from 0 to 9.


Each image is a grayscale image of size 28x28 pixels.

**Files**


train.csv: Contains the training data for the model. Each row represents an image, with pixel values as features and the corresponding digit as the label.


test.csv: Contains the test data for evaluating the model. Each row represents an image with pixel values but without a label.



**Project Structure**


Handwritten Digtal Recognization.ipynb: Jupyter Notebook containing the code for training and evaluating the model.


train.csv: Training dataset containing labeled handwritten digits.


test.csv: Test dataset containing unlabeled handwritten digits for prediction.

**Usage**

Download the dataset from Kaggle and place the train.csv and test.csv files in the project directory.


Open the Handwritten Digtal Recognization.ipynb notebook.


Run the cells in sequence to train the model and evaluate its performance.

**Model Description**


The model used in this project is a neural network designed to classify the digits based on pixel data from the images. The key steps include:


1. **Data Preprocessing:**
 
  
    Normalize the pixel values to a range of [0, 1].
  
  
    Split the data into training and validation sets.

2. **Model Architecture:**
 
  
   A neural network model with multiple dense layers is used.
  
  
    Activation functions like ReLU (Rectified Linear Unit) are used for non-linear transformation.
  
  
   Softmax activation function is used for the output layer to classify the digits.

3. **Training:**


   The model is compiled using categorical cross-entropy loss and optimized using the Adam optimizer.
  
  
   Model performance is evaluated using accuracy on the validation data.

4. **Prediction:**

     After training, the model is used to predict the digits in the test dataset.

**Results**


  The model achieves an accuracy of approximately  85% on the test set, which shows its effectiveness in recognizing handwritten digits.

**Conclusion**


  This project demonstrates how machine learning can be used to recognize handwritten digits with high accuracy. With proper training, the model is capable of identifying digits in unseen data.


