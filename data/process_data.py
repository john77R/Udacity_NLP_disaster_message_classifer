# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:47:23 2021

@author: John
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load data returns a dataframe"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    
    return df


def clean_data(df):
    """Function to clean text data"""
  
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    
    # remove last 2 characters of catergory names
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
        # convert column from string to numeric
        #categories[column] = pd.to_numeric(categories[column], downcast="integer")
        categories[column] = categories[column].astype(int)
    
    categories.drop(categories[categories['related'] > 1].index, inplace = True)
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],join='inner',axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(subset=['message'])
    
    return df

def save_data(df, database_filename):
    """save dataframe as sql database file"""
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')  


def main():
    """Function to call all sub functions  to process and clean data."""
    
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
