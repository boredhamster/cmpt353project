import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('reddit extracter').config("spark.ui.showConsoleProgress", "true").getOrCreate()

reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-subset-2021-2023'

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

def main():
    reddit_submissions = spark.read.json(reddit_submissions_path, schema=submissions_schema)
    reddit_comments = spark.read.json(reddit_comments_path, schema=comments_schema)
    
    subs = ['vancouver']
    
    reddit_submissions.where(reddit_submissions['subreddit'].isin(subs)) \
        .where((functions.col('year') >= 2021) & (functions.col('year') <= 2023)) \
        .sample(fraction=0.1) \
        .write.json(output + '/submissions', mode='overwrite', compression='gzip')
    reddit_comments.where(reddit_comments['subreddit'].isin(subs)) \
        .where((functions.col('year') >= 2021) & (functions.col('year') <= 2023)) \
        .sample(fraction=0.1) \
        .write.json(output + '/comments', mode='overwrite', compression='gzip')
    

main()