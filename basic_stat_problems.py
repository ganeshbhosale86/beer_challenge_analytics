import pandas as pd
import matplotlib.pyplot as plt
import utils

def top_breweries_by_ABV(df, num):
    names = df.loc[ df[ 'beer_ABV' ] > df[ 'beer_ABV' ].mean()] \
        .sort_values( by = [ 'beer_ABV' ], ascending = False )['beer_brewerId']
    topBrewers = []
    for name in names:
        if len(topBrewers) == num:
            break
        if name not in topBrewers:
            topBrewers.append(name)
    #print(topBrewers)
    return topBrewers

def sort_map(map):
    map = {key: val for key, val in sorted(map.items(), key = lambda ele: ele[1], reverse=True)}         
    return map

def year_with_highest_ratings(df):
    dates = df.loc[ df[ 'review_overall' ] == 5] \
        ['review_time']
    #print(len(dates))
    yearRatingDict = {}
    for date in dates:
        if date.year in yearRatingDict:
            yearRatingDict[date.year] = yearRatingDict.get(date.year) + 1
        else:
            yearRatingDict[date.year] = 1
    yearRatingDict = sort_map(yearRatingDict)   
    return yearRatingDict

def plot_date_ratings_bar(year_rating_dict):
    plt.bar(*zip(*year_rating_dict.items()))
    plt.xlabel('Year')
    plt.ylabel('No of Reviews')
    plt.title('Per Year Highest Review Ratings')
    plt.show()


def recommend_top_beers(df, num):
    no_of_riewies = df.loc[ df[ 'review_overall' ] == 5].groupby('beer_beerId')[['beer_beerId','review_time']].apply(lambda x: x.values.tolist())
    #print(no_of_riewies.shape)
    review_dict = {}
    for beer_id in no_of_riewies:
        review_dict[beer_id[0][0]] = len(beer_id)
    review_dict = sort_map(review_dict)    
    return review_dict

def plot_top_beer_review_ratings(review_dict):
    top_reviews = {}
    for x in list(review_dict)[0:10]:
        #print ("key {}, value {} ".format(x,  review_dict[x]))
        top_reviews[review_dict[x]] = x
    plt.bar(*zip(*top_reviews.items()))
    plt.xlabel('Beer Id')
    plt.ylabel('No Of reviews')
    plt.title('Top Beer Review Ratings')
    plt.show()

def plot_top_beer_styles(beer_style_dict):
    top_style = {}
    for x in list(beer_style_dict)[0:3]:
        #print ("key {}, value {} ".format(x,  review_dict[x]))
        top_style[x] = beer_style_dict[x]
    plt.bar(*zip(*top_style.items()))
    plt.title('Top Beer Style as per Review Ratings')
    plt.show()    

def favourite_beer_style(df):
    beer_styles = df.loc[df[ 'review_overall' ] == 5].groupby('beer_style')[['beer_style','review_time']].apply(lambda x: x.values.tolist())
    beer_style_dict = {}
    for beer_style_obj in beer_styles:
        beer_style_dict[beer_style_obj[0][0]] = len(beer_style_obj)
    beer_style_dict = sort_map(beer_style_dict)
    #print(beer_style_dict)
    plot_top_beer_styles(beer_style_dict)
    return list(beer_style_dict.keys())[0]

# Read Beer Challenge CSV
df = utils.read_beer_challenge_csv()

# Perform All stats
utils.create_and_plot_stats(df)

# Top 3 (or passed number) breweries with strongest beer (based on max ABV value)
num = 3
top_brewaries = top_breweries_by_ABV(df, num)
print("Top {} breweries with strongest beer :: ".format(num), top_brewaries)

# Year with highest ratings
year_rating_dict = year_with_highest_ratings(df)
print("Year with Highest Rating ::", list(year_rating_dict.keys())[0])
plot_date_ratings_bar(year_rating_dict)

# Recommend Top 3 (or num) beers
num = 3
review_dict = recommend_top_beers(df, num)
print("Top {} Recommended Beers :: ".format(num), list(review_dict.keys())[:num])
#print(len(review_dict))
plot_top_beer_review_ratings(review_dict)
  
# Favourite beer style
beer_style = favourite_beer_style(df)
print("Favourite Beer style based on reviews :: ", beer_style)

