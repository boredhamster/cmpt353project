import sys
from pyspark.sql import SparkSession, functions, types, Row
import contractions

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


def lowercase(string):
    return string.lower()

lowercase_udf = functions.udf(lowercase, types.StringType())

def fix_contractions(string):
    return contractions.fix(string)

fix_contractions_udf = functions.udf(fix_contractions, types.StringType())


def main(inputs):
    
    reddit_submissions_data = spark.read.json(inputs + '/submissions', schema=submissions_schema)
    reddit_comments_data = spark.read.json(inputs + '/comments', schema=comments_schema)
    
    
    # Remove submissions/comments with no text
    
    remove_text = ['', '[deleted]', '[removed]']
    
    reddit_submissions_data = reddit_submissions_data.where(~functions.col('selftext').isin(remove_text))
    reddit_comments_data = reddit_comments_data.where(~functions.col('body').isin(remove_text))
    
    # Remove contractions (e.g. don't -> do not, I'm -> I am)
    
    reddit_submissions_data = reddit_submissions_data.withColumn('selftext_no_contraction',
                                    fix_contractions_udf(reddit_submissions_data['selftext'])) \
                                    .drop('selftext')
                                    
    reddit_submissions_data = reddit_submissions_data.withColumnRenamed('selftext_no_contraction',
                                                                        'selftext')
    
    reddit_comments_data = reddit_comments_data.withColumn('body_no_contraction',
                                    fix_contractions_udf(reddit_comments_data['body'])) \
                                    .drop('body')
                                    
    reddit_comments_data = reddit_comments_data.withColumnRenamed('body_no_contraction',
                                                                  'body')
    
    
    reddit_submissions_data.write.json(inputs + '-submissions', mode='overwrite', compression='gzip')
    reddit_comments_data.write.json(inputs + '-comments', mode='overwrite', compression='gzip')
    
    
inputs = sys.argv[1]
assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')

main(inputs)