import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        load data into the dataframe
        Input:
            messages_filepath: filepath for the messages
            categories_filepath: filepath for the categories
        Output:
            df: combined dataset
    '''
    # load messages dataset
    messages = pd.read_csv("messages.csv")
    # load categories dataset
    categories = pd.read_csv("categories.csv")
    # merge datasets
    df = messages.merge(categories, on = 'id')
    
    return df

def clean_data(df):
    '''
        clean the data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # rename the columns of `categories`
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    #Replace 2 with 1
    df.replace(2, 1, inplace=True)
    # drop duplicates
    df = df[df.duplicated()==False]
    return df
    

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesWithCategories', engine, index=False)


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