# Disaster-Response-Project
Classify Responses during disasters into 36 different Categories. THe web app allows you to input a message which will auto classify it into the 36 categories. Part of Udacity Data Scientist Nanodegree Term 2 Project 2


##### Table of Contents  
- [Project Motivation](#project-motivation)
- [How to run the scripts](#how-to-run-the-scripts)
- [Libraries used:](#libraries-used-)
- [Files in the Repository](#files-in-the-repository)
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
1. Messages.csv - Raw Seattle listings data as of February 09, 2019 provided by Inside Airbnb team. Data can be found [here](http://insideairbnb.com/get-the-data.html)
2. Categories.csv - Raw Boston listings data as of February 09, 2019 provided by Inside Airbnb team. Data can be found [here](http://insideairbnb.com/get-the-data.html)
3. Classifier.pkl - Trained Model that can be used to classify any given message amongst the 36 categories.

## Acknowledgements 
1. [Figure Eight](https://www.figure-eight.com/) for providing us the data.
2. Udacity for giving the direction and scope of the project
