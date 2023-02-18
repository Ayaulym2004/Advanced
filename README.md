# Advanced
[Final_project.pdf](https://github.com/Ayaulym2004/Advanced/files/10774051/Final_project.pdf)

Prepared by Ormanova Ayagul and Zhailmina Ayaulym   BDA-2101
Report
GitHub account: https://github.com/Ayaulym2004/Advanced.git
YouTube Video: https://youtu.be/Rib8QQWBFI4

1. Introduction
1.1 Problem
With the development of technology, more and more books and documents in the field of sciences are moving to digital format. Mathematics is widely used in many fields of science, such as physics, engineering, medicine and economics. One of the key tasks is the analysis and understanding of digital documents. Optical Character Recognition (OCR) is used to improve the accuracy of character and number recognition in electronic books. However, the recognition of handwritten mathematical expressions is still a very difficult task in the field of computer vision.
CNN, a type of deep learning model, is specifically designed to process data with a grid pattern, like images, by mimicking the organization of the visual cortex in animals. Developing a reliable handwritten equation solver using CNN is a challenging image processing task, especially when it comes to recognizing handwritten mathematical expressions. 
In this study, we focus on the recognition of quadratic equations, using a convolutional neural network to extract objects from an image and subsequent symbol processing to solve the equation. However, the main problem is the distinction between handwritten tasks in the image and the correct determination of the conclusion of the equation upon detection. When segmenting an image, the square terms can be similar to other curves and numbers in the horizontal projection of the equation. We solve this problem by compact horizontal projection and using related components to correctly analyze symbols such as the "+" sign.
1.2 Literature review with links (another solutions)
Extracting Features. For optimal contour extraction results, it is recommended to first invert the image so that the object appears white and the background appears black. Then, converting the image to a binary format would further enhance the quality of the output.
Pradeep and colleagues (2010) proposed a feed-forward neural network method that utilizes diagonal, horizontal, and vertical features to categorize handwritten characters. The technique involves a diagonal feature extraction approach.
There are many projects that use an online approach to convert handwritten mathematical expressions into text equivalents, such as TEX or MathML. In these articles, you can also find projects that provide a user interface that provides feedback and makes it possible to quickly fix errors, which will make the use of the project more convenient for the end user.
The article (Zanibi) describes a mathematical expression recognition scheme (MER) that uses SVN and a projection histogram to recognize simple expressions. This scheme is part of an autonomous system for recognizing handwritten expressions. The article also discusses a variety of methods for extracting and recognizing features. Zabini and his colleagues in 2002 proposed an efficient and reliable system for recognizing both printed and handwritten mathematical symbols.
1.3 Current work (description of the work)
Import Data.
•	tensorflow 
• tensorflow-gpu
•	cv2
•	matplotlib 
•	tensorflow-datasets 
•	ipywidgets
•	numpy
Once the necessary packages are installed, the code imports the TensorFlow library and allows you to increase the amount of memory for any available GPUs in the system. This will allow TensorFlow to allocate GPU memory according to needs instead of allocating it all in advance.
In addition, the code imports the package "tensorflow_datasets" and uses it to extract the data set. The dataset consists of images of all the digits in various
types of writing, as well as addition and subtraction signs with dimensions of 30x30. It is divided into two groups, one for training and the other for testing.
Construct Training data. 
To train the data using a convolutional neural network (CNN), the dataset needs to be reshaped since CNNs work with two-dimensional data. The labels column is first assigned to the variable y_train, then the labels column is dropped from the dataset, and the remaining data is reshaped to 30 by 30, making it suitable for the CNN.
To fit CNN to the data, the code uses the provided lines of code. After the training is completed, the resulting model can be saved as a JSON file, which can be used for future forecasts without having to retrain the model every time for three hours. The following code can be used to save the trained model as a JSON file.
![image_2023-02-18_17-46-07](https://user-images.githubusercontent.com/125453394/219864239-73602ee1-7952-44b9-ae1c-5ddc0b53690f.png)
Construct Testing data. 
To use the trained model to solve handwritten equations, the first step is to import the saved model using the provided line of code. Next, an input image containing a handwritten equation is needed. The image is then converted to a binary image and inverted if the digits or symbols are in black. The code uses the 'findContour' function to obtain the contours of the image, with the contours obtained from left to right by default.
The bounding rectangle for each contour is then obtained, and if there are multiple contours for the same digit or symbol, the code checks if the bounding rectangles overlap. If they do overlap, the smaller rectangle is discarded. The remaining bounding rectangles are then resized to 30 by 30.
Using the trained model, the corresponding digit or symbol for each bounding rectangle is predicted and stored in a string. The 'eval' function is then used on the string to solve the equation.

2. Data and methods 
Dataset downloaded from kaggle, it contains ~100.000 handwritten math symbols, but for our problem we choosed only few symbols (0-10, +, - and *), because some of them are very similar (for example 1 and /)
This is a example of set of Sigma, a lot of pictures with the same symbol but written in different ways, with inaccuracies, etc.
![image](https://user-images.githubusercontent.com/125453394/219864257-77d550ff-766e-48a3-a6b1-d276fc262684.png)
For our case, we chose Sequential model with 4 layers, where the first layer is the input with dimensions 30*30=900 (for pixel-by-pixel data inputs into the model), the second and third layers are intermediate with sizes 128 and 64. The last layer is output, the size of the number of characters we need.
![image](https://user-images.githubusercontent.com/125453394/219864270-8b99b017-3a1a-45dd-a18d-48365d8e569b.png)
The sequential model is a good option for static output, since when we get the prediction result, we will get an array with str and float, where the first is our category (symbol), and float is the probability of a match

3. Results
We achieved the best result with 10 epochs and a test data size of 1\2 of the total number of data. Thus the result is:
loss: 0.2998 
accuracy: 0.9144 
val_loss: 0.7401 
val_accuracy: 0.8603
![image](https://user-images.githubusercontent.com/125453394/219864296-9f04fcfc-027b-4b46-8ea7-760601a6ad0f.png)
![image](https://user-images.githubusercontent.com/125453394/219864298-aceea547-e96b-40d6-b9c6-9e624df71ad1.png)
To fully test the work, we wrote a script that will use our model to determine the numerical expression and calculate the answer (for example, 2+3)
The logic and operation of the script is:
1) Getting an image and adding a black and white filter 
2) definition of shaded areas and their separation from each other, definition of erroneous areas
3) running each section through the model
4) composing an expression as a string and solving via eval()
![image](https://user-images.githubusercontent.com/125453394/219864304-30dd1d62-ccf3-429a-ad11-49c42a5a97af.png)

4. Discussion
4.1 Critical review of results
The resulting model demonstrates recognition accuracy of about 90% on the test sample, which is a good result. Visual error analysis showed that the main errors are related to the displacement or inclination of handwritten characters, as well as errors in data markup. In general, the resulting model can be used for handwritten character recognition in applications related to optical character recognition or handwritten data analysis.
4.2 Next steps
Opportunities and development options
Opportunities for the development of this direction include the following:
Data set expansion. Adding more character images and more diverse handwriting patterns can improve the accuracy of the model.
Using generative networks (GANS) to generate additional data. Generative networks can create new images based on existing ones, which can help increase the size of the dataset.
Using transfer learning. Using pre-trained models to classify handwritten characters can speed up the learning process and improve the accuracy of the model.
Development of a model for handwriting recognition. A model that can recognize handwritten text can have wide applications in the field of natural language processing, including automatic recognition of signatures and handwritten documents.
Using a model for automatic digitization of handwritten mathematical expressions. The model can be used to recognize handwritten mathematical symbols, which can be useful for automatic digitization of mathematical formulas in scientific papers or in educational applications.
Developing a model for recognizing other languages. The model can be trained to recognize handwritten characters in other languages, which may have practical application for processing handwritten documents in different languages.
In general, this direction has wide opportunities for development and application in various fields. Improving the accuracy of the model, expanding the data set and using other machine learning techniques can help in creating more accurate and efficient models for recognizing handwritten characters and text.

References:
Vipul G. (2019). Handwritten Equation Solver using Convolutional Neural Network 
https://vipul-gupta73921.medium.com/handwritten-equation-solver-using-convolutional-neural-network-a44acc0bd9f8
https://www.geeksforgeeks.org/handwritten-equation-solver-in-python/
Y.A. Joarder (2018). Recognition and Solution for Handwritten Equation Using Convolutional Neural Network
(PDF) Recognition and Solution for Handwritten Equation Using Convolutional Neural Network (researchgate.net)

