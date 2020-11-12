import pandas as pd
import datetime
import matplotlib.pyplot as plt

def read_beer_challenge_csv():
    df = pd.read_csv('resource/BeerDataScienceProject.csv',encoding='latin1',)
    df['review_time'] = pd.to_datetime(df['review_time'], unit='s')
    #print(df.head(5))
    #print(df.shape)
    #print(df['review_time'].dtype)
    return df

def check_missing_values(df):
    print(df.isna().sum())

def drop_na_rows(df):
    print("Before Dropping NaN Values ::",df.shape)
    df = df.dropna()
    print("After Dropping NaN Values ::",df.shape)  
    return df  

def check_and_remove_duplicate_rows(df):
    updated_df = df[ [ 'beer_beerId', 'review_profileName', 'beer_style', 'review_time', 'review_text' ] ].drop_duplicates()
    print("Total Rows", df.shape)
    print("Unique Rows", updated_df.shape)
    

def print_unique_stats(df):
    print( 'Unique beers:', df[ 'beer_beerId' ].nunique())
    print( 'Unique breweries:', df[ 'beer_brewerId' ].nunique())
    print( 'Unique users:', df[ 'review_profileName' ].nunique())
    print( 'Unique beer styles:', df[ 'beer_style' ].nunique())
    return df

def create_and_plot_stats(df):
    pass













