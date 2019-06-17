import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Input - Filepath to messages & categories file in a csv format
    Output - A dataframe that combines both messages and categories data on the id column
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on='id')
    return df

def clean_data(df):
    '''Input - A dataframe with a column names categories
    Output - A cleaned data frame that undergoes the following transformations
    1. Split it's categories column on ';' character
    2. Pull the column headers based on the values in the cell
    3. Extract the actual last character (0 or 1) in each cell
    4. Drop duplicates in the dataframe
    5. Remove any other values in target variables apart from 0 or 1
    '''
    df_categories = df['categories'].str.split(";",expand=True) # split categories column by ';' character
    
    category_colnames = df_categories.iloc[0].apply(lambda x: x[:-2]) # get column names for target variables
    df_categories.columns = category_colnames
    
    for column in df_categories:
        df_categories[column] = df_categories[column].apply(lambda x:x[-1]) # set each value to be the last character of the string
        df_categories[column] = df_categories[column].astype(int) # convert column from string to numeric
    
    df = df.drop(columns='categories')
    df = pd.concat([df,df_categories],axis=1)
    
    df.drop_duplicates(inplace=True) # drop duplicates
    df = df[df.related!=2] #drop values where value is 2 (we need only 0s or 1s) 
    
    return df
    
def save_data(df, database_filepath):
    ''' Creates a SQL database table 'msg_categories' using the input Dataframe at the given database
    Input - A dataframe and the database filepath 
    Output - None
    '''    
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('msg_categories', engine, index=False, if_exists='replace')  


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