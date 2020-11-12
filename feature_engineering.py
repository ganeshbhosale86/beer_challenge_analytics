import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import utils

def find_correlation(df):
    print(df.corr())
    ## Pearson Correlation
    sns.heatmap( df.corr(), center = 0,  vmin = -1, vmax = 1 )
    plt.title( 'Pearson Correlation' )
    plt.show()
    ## Spearman Correlation
    sns.heatmap( df.corr( method = 'spearman' ), center = 0,  vmin = -1, vmax = 1 )
    plt.title( 'Spearman Correlation' )
    plt.show()

# Read CSV file
df = utils.read_beer_challenge_csv()

# Standard Correlation between features
find_correlation(df)

# Feature Samples Training Set 
X = df[ [ 'review_palette', 'review_aroma', 'review_appearance', 'review_taste' ] ]
y = df[ 'review_overall' ] 

# Linear regression model
linear_model = LinearRegression( normalize = True )
linear_model.fit( X, y)
preds = linear_model.predict(X)

# Correlation of Coefficients for feature samples 
# From observation, the review taste and review aroma has significant impact on overall score 
print("Coefficient values for features :: ", linear_model.coef_)
print("Model Score :: ", linear_model.score(X,y))

