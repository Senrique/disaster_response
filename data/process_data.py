import sys
import pandas as pd
from sqlalchemy import create_engine
import os


def load_data(messages_filepath, categories_filepath):
    """
    Load the data
    
    Arguments:
      messages_filepath(string): the file path of messages.csv
      categories_filepath(string): the file path of categories.csv
      
    Return:
      df: merged dataframe of messages + categories
    """
	messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id",how='left')
    return df


def clean_data(df):
    """
    Process the data:
	1. Split the categories column in to separate categories columns
	2. Rename those columns with appropriate category names
	3. Convert the categorical values to integer values
	4. Remove duplicates (if any)
    
    Arguments:
      df(dataframe): merged dataframe from load_data function
      
    Return:
      df: processed dataframe
    """
    # Split the categories column in to separate categories columns
	categories = df['categories'].str.split(pat=";",expand=True)
    # Obtain the category names
	row = categories.iloc[0]
    category_colnames = [x.split("-")[0] for x in row]
    # Rename category columns
	categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x.split("-")[1] for x in categories[column]]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('float').astype('Int64')
    # Drop duplicates
	df = df.drop(["categories"],axis=1)
    df = pd.concat([df, categories], axis=1, sort=False)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save the processed dataframe into SQLite database
    
    Args:
        df(Dataframe): Processed dataframe
        database_filename(string): the file path to save the .db file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()