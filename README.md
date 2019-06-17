# Disaster-Response-Project
Classify Responses during disasters into 36 different Categories. THe web app allows you to input a message which will auto classify it into the 36 categories. Part of Udacity Data Scientist Nanodegree Term 2 Project 2
Github link: https://github.com/yvsajay/Disaster-Response-Project

##### Table of Contents  
- [Project Motivation](#project-motivation)
- [How to run the scripts](#how-to-run-the-scripts)
- [Libraries used:](#libraries-used-)
- [Files in the Repository](#files-in-the-repository)
- [The ETL Process](#the-etl-process)
- [The Model](#the-model)
- [The Web app](#the-web-app)
- [Acknowledgements](#acknowledgements)


## Project Motivation
>This project is being done as a part of the Udacity's Data Scientist Nano Degree program Term 2. 
>The questions I intended to answer 

## How to run the scripts
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Libraries used:
1. [Numpy](https://www.numpy.org/)
2. [Pandas](https://pandas.pydata.org/)
3. [Plotly](https://plot.ly/python/)
4. [NLTK](https://www.nltk.org/)
5. [SQLAlchemy](https://www.sqlalchemy.org/)
6. [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)

## Files in the Repository
1. Messages.csv - Raw messages data
2. Categories.csv - Categories for the messages. 
3. Classifier.pkl - Trained Model that can be used to classify any given message amongst the 36 categories.

## The ETL Process
In a Python script, process_data.py will do the below steps
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

## The Model
In a Python script, train_classifier.py, write a machine learning pipeline that:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file 

## The Web App
The app will help to determine any input message into either of the 36 categories. It also has a couple of visualizations on the training data. One, is the count of messages by their "Genre", and the other, the count of messages by their "Category" identified.

## Acknowledgements 
1. [Figure Eight](https://www.figure-eight.com/) for providing us the data.
2. Udacity for giving the direction and scope of the project
