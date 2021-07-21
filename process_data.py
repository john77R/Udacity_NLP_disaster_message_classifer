import sys
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
     """
    Load data from a file location
    Input: messages_filepath 
            path to the messagages data csv location
            
           categories_filepath
            path to the categories data csv location
            
    Output: df = DataFrame
            Merged DataFrame of message and categories data
    """
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")


def clean_data(df):
     """
    Extracts categories and flags from categories data, remove duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row =categories.iloc[1:2]

    # use this row to extract a list of new column names for categories.
    # apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = pd.DataFrame(row.applymap(lambda x: (str(x))[:-2]))
    category_colnames = category_colnames.values.tolist()
    
    category_colnames = categories.columns
    categories=categories.replace(to_replace=r'\D', value='', regex=True)
    
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(subset=['message'])
    
    return df



def save_data(df, database_filename, Db_name = 'DisasterResponceDatabase'):
    
    """
    Save cleaned data to database
    Input: df - DataFrame from clean_data
           database_filename - Database file location of where post ETL data is to be stored
           Db_name - Can be any name to given to our Loaded Database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, index=False)  
    print("Data was saved to {} in the {} table".format(database_filename, Db_name))
                                


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