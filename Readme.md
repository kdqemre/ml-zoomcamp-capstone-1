Capstone Project-1  
Classiffiying the fast food images.
Emre Ozturk 
 
## Description of the problem:  
Using data Fast Food Classification Dataset - V2 we will classiffy the fast foods.
We have 10 classes.

 Baked Potato': 0,
 'Burger': 1,
 'Crispy Chicken': 2,
 'Donut': 3,
 'Fries': 4,
 'Hot Dog': 5,
 'Pizza': 6,
 'Sandwich': 7,
 'Taco': 8,
 'Taquito': 9}

Train : Contains 15,000 training images, with each class having 1,500 images.

Valid : Contains 3,500 validation images, with each class having 400 images.

Test : Contains 1,500 validation images, with each class having 100/200 images.

Considering training time i used small version of this data.

 
## Preprocessing:
IMPORTANT NOTE: Remember Input data also needs preprocessing for to use model and get correct predictions. 

## Modelling

    I used transfer learning and pre trained `Xception` algorithm with `imagenet`.
And i trained it for my classes.

While modeelling tried different:
- image sizes
- learning rates
- inner layers
- dropouts
- augmentations

All this process can be found projects notebook.

 
## Instructions (How to run the project):

First of all you need Python in your computer.
In the project folder you will find train.py , predict.py , trained model file and project’s notebook.
The trained algorithm needs a fast foods image as an input. Remember Input data also needs preprocessing .
-	If you want to use the project locally, load and use the pre-trained model file.

The trained model needs a water pump’s features as an input
-	If you want to tune the model and use different algorithms you can use train.py script
-	In the project's notebook you can see pre-processing and modelling.
-	If you want to use the model as a web service, you can use predict.py script. You should send a post request as an input and have to be json file.  
-   You can find predic-test.py script in project's folder for test the model
-   You will find a Docker file in the folder.  

## Dependencies   (How to run the project)
For use the project you need some dependencies:
- First of all you need Python in your computer.
- For notebook you need an anaconda distribution and jupyter notebook.
- tensorflow: You need to install this library for load and use the model. It is essential for use the pre trained model.
- If you want to use the project’s notebook  you need to install some libraries:
Numpy , pandas, tensorflow.

-`requriments.txt` file includes all dependincies.

## How to install dependencies (for windows):

installing all dependincies with requirements.txt 
in your enviroment
`pip install -r requirements.txt`

First in cmd you can create a virtual environment:

conda create -n my-env
Than activate the environment:

conda activate my-env
Installing Anaconda distribution: You can download anaconda from here https://www.anaconda.com/products/individual

Installing Numpy:

You can use 

conda install numpy
or  
pip install numpy

Installing Pandas:

In virtual environment:

conda install pandas

or

pip install pandas

Installing tensorflow:

In your virtual environment:


pip install tensorflow



#Using Docker File:

First install the python 3.9, the slim version have less size
FROM python:3.9-slim

We have created a directory in Docker named app and we're using it as wor directory 
WORKDIR /app                                                                

Copy the requirements.txt into our working derectory 
COPY ["requirements.txt","./"] 

install the requirements.txt  dependecies we had from the project and deploy them 
RUN pip install -r requirements.txt

Copy any python files and the model we had to the working directory of Docker 
COPY ["predict.py","xception_v4_1_01_0.676.h5", "./"] 

We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

If we run the Docker image, we want our pump status prediction app to be running
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]









