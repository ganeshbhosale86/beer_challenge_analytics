# beer_challenge_analytics
Analytics on beer review dataset

Used libraries:
Python 3.7, Tensorflow2.3, spacy, keras, numpy, pandas, sklearn, matplotlib, seaborn, datetime 

Install all of the above libraries using pip install command. 
We can also create an enviornment where we can have different versions of these libraries like conda enviornment.
  1. We can install Anaconda.
  2. Create enviornments other than base (this is also fine). 
  3. Set all the paths 
  4. conda init - initialize the enviornment
  5. conda activate the environment  i.e. conda activate tf
  
Project structure:
  preprocessing.py   - basic text processing using spacy, create vector sequences, create word indexes, etc.
  utils.py      - Read csv file using pandas, perform some basic stats, etc.
  basis_stat_problems.py    - all the problems which require only statistics solved here (top 3 Breweries, highest rating year, favorite beer style, top 3 beer recommendation)
  feature_enginnering.py   - correlation between the features using sklearn
  sentiment.py   - sentiment model training using deep learning and compare with the rating score
  similarity.py   - find similarity between the reviews and cluster the drinkers accordingly
  resource
     stopwords.txt
     beer_challenge.csv  [not able to upload due to size]

Execute:
  make sure all of the above libraries install in an enviornment
  execute the py file as per the project structure description for example to run the stat problems do py basic_stats_problems.py
 
Please refer beer_challenge_approach.docx for the in detail overall approach and algorithms.  
