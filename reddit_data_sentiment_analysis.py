import sys
assert sys.version_info >= (3, 10) # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


comments_schema = types.StructType([
    types.StructField('body', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

submissions_schema = types.StructType([
    types.StructField('num_comments', types.LongType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])


def main(input_submissions, input_comments, output):
    
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)
    
    reddit_submissions_data = reddit_submissions_data.withColumn('id', functions.monotonically_increasing_id())
    reddit_comments_data = reddit_comments_data.withColumn('id', functions.monotonically_increasing_id())
    
    
    training_data = load_dataset('sentiment140', trust_remote_code=True)['train']
    training_data = training_data.to_pandas()
    training_data = training_data.sample(frac=0.01)
    
    X = training_data['text']
    y = training_data['sentiment']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    model_svc = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words='english', 
                        max_df=0.25, max_features=10000),
        LinearSVC(penalty='l1', C=0.05, dual=False)
    )
    
    model_svc.fit(X_train, y_train)
    
    print(f"Model's training score on the Twitter data: {model_svc.score(X_train, y_train)}")
    print(f"Model's validation score on the Twitter data: {model_svc.score(X_valid, y_valid)}")
    
    
    reddit_submissions_text = reddit_submissions_data.select('selftext').toPandas()
    
    submissions_predictions = model_svc.predict(reddit_submissions_text['selftext'])
    submissions_predictions = pd.DataFrame(submissions_predictions, columns=['sentiment'])
    submissions_predictions = spark.createDataFrame(submissions_predictions)
    submissions_predictions = submissions_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions = reddit_submissions_data.join(submissions_predictions, on='id').drop('id')
    
    
    reddit_comments_text = reddit_comments_data.select('body').toPandas()
    
    comments_predictions = model_svc.predict(reddit_comments_text['body'])
    comments_predictions = pd.DataFrame(comments_predictions, columns=['sentiment'])
    comments_predictions = spark.createDataFrame(comments_predictions)
    comments_predictions = comments_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments = reddit_comments_data.join(comments_predictions, on='id').drop('id')
    
    
    reddit_submissions.write.json(output + '-submissions', mode='overwrite', compression='gzip')
    reddit_comments.write.json(output + '-comments', mode='overwrite', compression='gzip')
    
    

input_submissions = sys.argv[1]
input_comments = sys.argv[2]
output = sys.argv[3]
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')
#sc = spark.sparkContext

main(input_submissions, input_comments, output)