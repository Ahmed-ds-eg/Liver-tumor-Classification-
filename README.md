#Liver Tumor Classification
This project aims to classify liver tumors into three classes: normal, malignant, and benign. We achieved 100% recall, precision, and harmonic mean in test data by using the ResNet50 model.

##Tools used:

* Keras
* TensorFlow
* NumPy
* os
##Image preprocessing:

* Resized the images to 512x512
* Applied non-local means filter
##To run the code:

Install the necessary libraries listed in the requirements.txt file. 
Run liver-tumor-classification-using-resnet50.ipynb to train and evaluate the model.
Note: The code assumes that the dataset is organized in the following structure:


data/
    train/
        normal/
            normal_001.jpg
            normal_002.jpg
            ...
        malignant/
            malignant_001.jpg
            malignant_002.jpg
            ...
        benign/
            benign_001.jpg
            benign_002.jpg
            ...
    test/
        normal/
            normal_101.jpg
            normal_102.jpg
            ...
        malignant/
            malignant_101.jpg
            malignant_102.jpg
            ...
        benign/
            benign_101.jpg
            benign_102.jpg
            ...




##Project Description
This project aims to classify liver tumors into three classes: normal, malignant, and benign. We used a ResNet50 model trained on a dataset of liver tumor images. To improve the performance of the model, we preprocessed the images by applying a non-local mean filter and resizing them to 512x512. We used Keras and TensorFlow libraries for training and evaluating the model, as well as NumPy and OS for data preprocessing.

##Dataset
The dataset used for this project was obtained from Kaggle. It contains 1,819 images of liver tumors, divided into three classes: normal, malignant, and benign. We split the dataset into training, validation, and testing sets with a ratio of 70%, 15%, and 15%, respectively.

##Model Architecture
We used a pre-trained ResNet50 model for this classification task. We added a dense layer with 3 units and softmax activation function as the output layer to classify the images into the three classes.

##Results
After training the model on the preprocessed dataset, we achieved 100% precision, recall, and F1-score on the testing set. The model was able to accurately classify liver tumors into their respective classes.

##Conclusion
In conclusion, we successfully developed a liver tumor classification model using ResNet50 that achieved high accuracy on the testing set. By using a non-local mean filter and resizing the images, we were able to improve the performance of the model. This model could potentially be used for diagnosing liver tumors in clinical settings.
