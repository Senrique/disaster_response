# Disaster Response Pipeline Project for Udacity

This is a project to classify the tweets sent during disaster events in to various categories so that they can be sent to an appropriate disaster relief agency. We have received the data from Figure Eight and has been labelled by them.

The model output is done via flask webapp which outputs the plausible category of the incoming tweet. The incoming tweet undergoes a data cleaning process via an ETL pipeline and then in to the model developed on the data set provided by Figure Eight. The tweet is then categorized in to one of the 36 categories available such as 'aid related', 'search and rescue', 'child alone' etc.

### Requirements
Software: Python 3
Packages: 
		Webapp: pandas, flask, plotly, nltk and sqlalchemy
        ETL and ML Pipelines: pandas, numpy and sklearn
Datasets:
		disaster_messages.csv: List of tweets related to disaster events
        disaster_categories.csv: Categorization of the tweets in disaster_messages.csv. A single tweet can be categorized in to multiple categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Github Repository:

You can find the repository on https://github.com/Senrique/disaster_response