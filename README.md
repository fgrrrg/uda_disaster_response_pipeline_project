# Disaster Response Pipeline Project

# Summary

This is a natual language processing machine learning pipeline which classifies (social-media) messages during an emergency situation.
The code in this repository has 3 functions:
* The ETL script reads the prelabelled disaster response messages, cleans it and saves it as an SQL database
* The ML script loads the data from the SQL db file, trains a classifier machine learning model, evaluates it and saves the model as a pickle file
* The webapp uses the pre-trained model to classify messages into the different categories and shows some statistics of the database

## Requirements
* numpy
* pandas
* flask
* scikit-learn
* NLTK
* plotly
* sqlalchemy
  
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Necessary files in the repository
* data
    * process_data.py
    * disaster_categories.csv
    * disaster_messages.csv
* models
    * train_classifier.py
* app
  * run.py
