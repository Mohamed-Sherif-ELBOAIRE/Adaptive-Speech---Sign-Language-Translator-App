# Adaptive-Speech---Sign-Language-Translator-App
> This repository contains the code for training a Convolutional Neural Network (CNN) model to recognize American Sign Language (ASL) gestures. The model uses TensorFlow and Keras libraries for building and training the CNN. Additionally, the code includes hyperparameter tuning using TensorBoard for optimizing the model's performance.
> Created AI Model to recognize images and can be transformed into text to the users of the Application

## Prerequisites

To run the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (cv2)

## Getting Started

Follow these steps to get started with the ASL Translator project:

1. Clone the repository to your local machine.

2. Ensure that the required libraries are installed by running:
``` pip install tensorflow numpy pandas matplotlib seaborn opencv-python ```
3. Download the ASL Alphabet Dataset and store the training and testing images in the appropriate directories
[Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Introduction
- One of the most important pillars on which our society is built is social communication. It is known that language is the only way to communicate and interact with each other verbally and nonverbally. People with special needs are also members of this society and have the right to communicate with the outside world simply and professionally. The application enables communication with deaf users and vice versa. Employing the English language as a medium of communication is the key feature of this application to learn all the sign language terms.

- Two aspects prove our application is powerful, the first one, and the ability of normal people to communicate with deaf people without learning or knowing sign language. This is achieved in the chat compartment by voice recognition of words or by typing the words in English. Secondly and more importantly, deaf people communicate with normal people by choosing sign images from various categories sorted in a database, as a result, the images are transformed into text. But our main constraint on users is that deaf people have been learned the English language. 

### American Sign Language (ASL) 
ASL is a complete, natural language that has the same linguistic properties as spoken languages, with grammar that differs from English. ASL is expressed by
movements of the hands and face. It is the primary language of many North Americans who are deaf and hard of hearing and is used by many hearing people as well.
No person or committee invented ASL. The exact beginnings of ASL are not clear, but some suggest that it arose more than 200 years ago from the intermixing of local sign languages and French Sign Language (LSF, or Langue des Signes Française). Today’s ASL includes some elements of LSF plus the original local sign languages

## Dataset Exploration
The dataset contains images of various ASL gestures, and we have pre-defined classes for each gesture. Visualized Images and Lables.

## Data Preprocessing
Data preprocessing is an essential step before training the CNN model. The code includes data augmentation to improve the generalization of the model. Data augmentation techniques like rotation, zoom, and horizontal flip are applied to the training images.

## Hyperparameter Tuning
We perform hyperparameter tuning using TensorBoard to optimize the CNN model. The hyperparameters tested include filter size and the number of nodes in the dense layer. TensorBoard is used for visualization and tracking the model's performance.
The best hyperparameters are selected based on the highest accuracy achieved during the tuning process.
After Tuning hyperparameters, the best Model was with 3 kernel size and Dense 512. 

## Model Training
With the best hyperparameters selected, we build and train the CNN model using TensorFlow and Keras. The model architecture consists of several Convolutional, MaxPooling, BatchNormalization, Dropout, and Dense layers.
### Saving and Loading the Model
Once the model is trained, it is saved to disk to integerate with the Adapatve Speech Application. 

## Model Evaluation
The trained model is evaluated on the test dataset to measure its accuracy and loss
A confusion matrix is generated to visualize the model's performance on the test dataset splitted from the train dataset
Plotted train, validation, loss, and val_loss according to epochs 
Plotted train, val accurary, and loss to epochs 
Predictied and Visualized the test dataset to actual values

## Conclusion
This repository contains a complete implementation of an ASL Translator using CNNs with hyperparameter tuning. The trained model achieves high accuracy in recognizing ASL gestures, making it suitable for various applications. The code also includes visualization techniques for analyzing the training process and the model's performance on unseen data.

Feel free to explore the code, contribute to the project, and utilize the ASL Translator model for sign language recognition tasks!

