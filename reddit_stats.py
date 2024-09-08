import sys
assert sys.version_info >= (3, 10)  # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import os
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt 
import seaborn 


comments_schema = types.StructType([
    types.StructField('body', types.StringType()),
    types.StructField('sentiment', types.LongType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

submissions_schema = types.StructType([
    types.StructField('num_comments', types.LongType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('sentiment', types.LongType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main(input_submissions, input_comments, output):

    # Initialize SparkSession
    spark = SparkSession.builder.appName('example code').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Read and filter data
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)

    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]

    #submissions plot

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    submissions_neg = reddit_submissions_data.where(functions.col('sentiment') == 0)
    submissions_pos = reddit_submissions_data.where(functions.col('sentiment') == 4)

    monthly_submissions_neg = submissions_neg.groupby('month').agg(functions.count('*').alias('total_subs_neg'))
    monthly_submissions_pos = submissions_pos.groupby('month').agg(functions.count('*').alias('total_subs_pos'))

    join_monthly = monthly_submissions_neg.join(monthly_submissions_pos, on='month').toPandas()

    seaborn.set()
    plt.figure(figsize=(14, 6))

    join_monthly = join_monthly.sort_values('month')

    plt.bar(join_monthly['month'] - 0.2, join_monthly['total_subs_neg'], width=0.4, label='Submissions Neg', align='center')
    plt.bar(join_monthly['month'] + 0.2, join_monthly['total_subs_pos'], width=0.4, label='Submissions Pos', align='center')

    plt.xlabel('Months')
    plt.ylabel('Num. of Submissions')
    plt.xticks(join_monthly['month'], month_names)
    plt.legend()
    plt.title('Num. of Positive and Negative Submissions by Month')
    
    plt.savefig(output + '/submissions_monthly.png')
    
    plt.close()
    

    #comments plot
    

    comments_neg = reddit_comments_data.where(functions.col('sentiment') == 0)
    comments_pos = reddit_comments_data.where(functions.col('sentiment') == 4)

    monthly_comments_neg = comments_neg.groupby('month').agg(functions.count('*').alias('total_coms_neg'))
    monthly_comments_pos = comments_pos.groupby('month').agg(functions.count('*').alias('total_coms_pos'))

    join_monthly = monthly_comments_neg.join(monthly_comments_pos, on='month').toPandas()
    plt.figure(figsize=(14, 6))

    join_monthly = join_monthly.sort_values('month')
    plt.bar(join_monthly['month'] - 0.2, join_monthly['total_coms_neg'], width=0.4, label='Comments Neg', align='center')
    plt.bar(join_monthly['month'] + 0.2, join_monthly['total_coms_pos'], width=0.4, label='Comments Pos', align='center')

    plt.xlabel('Months')
    plt.ylabel('Num. of Comments')
    plt.xticks(join_monthly['month'], month_names)
    plt.legend()
    plt.title('Num. of Positive and Negative Comments by Month')
    
    plt.savefig(output + '/comments_monthly.png')
    
    plt.close()

    #end plt

    

    # Chi-squared Test (4-seasons)

    summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    fall_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(fall_months))
    winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))
    spring_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(spring_months))

    summer_submissions_neg = summer_submissions_filter.where(functions.col('sentiment') == 0)
    summer_submissions_pos = summer_submissions_filter.where(functions.col('sentiment') == 4)
    
    fall_submissions_neg = fall_submissions_filter.where(functions.col('sentiment') == 0)
    fall_submissions_pos = fall_submissions_filter.where(functions.col('sentiment') == 4)

    winter_submissions_neg = winter_submissions_filter.where(functions.col('sentiment') == 0)
    winter_submissions_pos = winter_submissions_filter.where(functions.col('sentiment') == 4)

    spring_submissions_neg = spring_submissions_filter.where(functions.col('sentiment') == 0)
    spring_submissions_pos = spring_submissions_filter.where(functions.col('sentiment') == 4)
    
    summer_sub_neg_counts = summer_submissions_neg.count()
    summer_sub_pos_counts = summer_submissions_pos.count()

    fall_sub_neg_counts = fall_submissions_neg.count()
    fall_sub_pos_counts = fall_submissions_pos.count()
    
    winter_sub_neg_counts = winter_submissions_neg.count()
    winter_sub_pos_counts = winter_submissions_pos.count()

    spring_sub_neg_counts = spring_submissions_neg.count()
    spring_sub_pos_counts = spring_submissions_pos.count()
    
    print(summer_sub_pos_counts, summer_sub_neg_counts, fall_sub_pos_counts, fall_sub_neg_counts, winter_sub_pos_counts, winter_sub_neg_counts, spring_sub_pos_counts, spring_sub_neg_counts)
    contingency_submissions = [[summer_sub_pos_counts, summer_sub_neg_counts],
                    [fall_sub_pos_counts, fall_sub_neg_counts],
                   [winter_sub_pos_counts, winter_sub_neg_counts],
                   [spring_sub_pos_counts, spring_sub_neg_counts] ]
    
    chi2_submissions = chi2_contingency(contingency_submissions)
    
    print('Chi-squared p-value submissions (4-seasons): ', chi2_submissions.pvalue)
    


    summer_comments_filter = reddit_comments_data.where(functions.col('month').isin(summer_months))
    fall_comments_filter = reddit_comments_data.where(functions.col('month').isin(fall_months))
    winter_comments_filter = reddit_comments_data.where(functions.col('month').isin(winter_months))
    spring_comments_filter = reddit_comments_data.where(functions.col('month').isin(spring_months))

    summer_comments_neg = summer_comments_filter.where(functions.col('sentiment') == 0)
    summer_comments_pos = summer_comments_filter.where(functions.col('sentiment') == 4)

    fall_comments_neg = fall_comments_filter.where(functions.col('sentiment') == 0)
    fall_comments_pos = fall_comments_filter.where(functions.col('sentiment') == 4)

    winter_comments_neg = winter_comments_filter.where(functions.col('sentiment') == 0)
    winter_comments_pos = winter_comments_filter.where(functions.col('sentiment') == 4)

    spring_comments_neg = spring_comments_filter.where(functions.col('sentiment') == 0)
    spring_comments_pos = spring_comments_filter.where(functions.col('sentiment') == 4)

    summer_com_neg_counts = summer_comments_neg.count()
    summer_com_pos_counts = summer_comments_pos.count()

    fall_com_neg_counts = fall_comments_neg.count()
    fall_com_pos_counts = fall_comments_pos.count()

    winter_com_neg_counts = winter_comments_neg.count()
    winter_com_pos_counts = winter_comments_pos.count()

    spring_com_neg_counts = spring_comments_neg.count()
    spring_com_pos_counts = spring_comments_pos.count()

    print(summer_com_pos_counts, summer_com_neg_counts, winter_com_pos_counts, winter_com_neg_counts)
    contingency_comments = [[summer_com_pos_counts, summer_com_neg_counts],
                [fall_com_pos_counts, fall_com_neg_counts],
               [winter_com_pos_counts, winter_com_neg_counts],
               [spring_com_pos_counts, spring_com_neg_counts]]

    chi2_comments = chi2_contingency(contingency_comments)
    
    print('Chi-squared p-value comments: (4-seasons)', chi2_comments.pvalue)
    
    
    # Plot of actual vs. expected counts of positive submissions (4-seasons)
    
    season_names = ['Summer', 'Fall', 'Winter', 'Spring']
    
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    
    temporary_x_ticks = np.array([0, 1, 2, 3])
    
    subs_pos_actual = [summer_sub_pos_counts, fall_sub_pos_counts, 
                       winter_sub_pos_counts, spring_sub_pos_counts]
    
    subs_pos_expected = chi2_submissions.expected_freq[:,0]
    
    plt.bar(temporary_x_ticks - 0.2, subs_pos_actual, width=0.4, label='Pos Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_pos_expected, width=0.4, label='Pos Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Positive Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Submissions')
    plt.legend()
    
    
    # Plot of actual vs. expected counts of negative submissions (4-seasons)
    
    plt.subplot(2, 1, 2)
    
    subs_neg_actual = [summer_sub_neg_counts, fall_sub_neg_counts, 
                       winter_sub_neg_counts, spring_sub_neg_counts]
    
    subs_neg_expected = chi2_submissions.expected_freq[:,1]
    
    plt.bar(temporary_x_ticks - 0.2, subs_neg_actual, width=0.4, label='Neg Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_neg_expected, width=0.4, label='Neg Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Negative Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Submissions')
    plt.legend()
    
    plt.savefig(output + '/actual_vs_expected_subs_4_seasons.png')
    plt.close()
    
    
    # Plot of actual vs. expected counts of positive comments (4-seasons)

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)

    coms_pos_actual = [summer_com_pos_counts, fall_com_pos_counts, 
                       winter_com_pos_counts, spring_com_pos_counts]

    coms_pos_expected = chi2_comments.expected_freq[:,0]

    plt.bar(temporary_x_ticks - 0.2, coms_pos_actual, width=0.4, label='Pos Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_pos_expected, width=0.4, label='Pos Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Positive Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Comments')
    plt.legend()


    # Plot of actual vs. expected counts of negative comments (4-seasons)

    plt.subplot(2, 1, 2)

    coms_neg_actual = [summer_com_neg_counts, fall_com_neg_counts, 
                       winter_com_neg_counts, spring_com_neg_counts]

    coms_neg_expected = chi2_comments.expected_freq[:,1]

    plt.bar(temporary_x_ticks - 0.2, coms_neg_actual, width=0.4, label='Neg Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_neg_expected, width=0.4, label='Neg Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Negative Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Comments')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output + '/actual_vs_expected_coms_4_seasons.png')
    plt.close()

    
    # Chi-squared Test (Summer (June-September) vs. Non-summer)
    
    summer_months = [6, 7, 8, 9]
    winter_months = [1, 2, 3, 4, 5, 10, 11, 12]
    
    summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))

    summer_submissions_neg = summer_submissions_filter.where(functions.col('sentiment') == 0)
    summer_submissions_pos = summer_submissions_filter.where(functions.col('sentiment') == 4)

    winter_submissions_neg = winter_submissions_filter.where(functions.col('sentiment') == 0)
    winter_submissions_pos = winter_submissions_filter.where(functions.col('sentiment') == 4)

    summer_sub_neg_counts = summer_submissions_neg.count()
    summer_sub_pos_counts = summer_submissions_pos.count()

    winter_sub_neg_counts = winter_submissions_neg.count()
    winter_sub_pos_counts = winter_submissions_pos.count()

    print(summer_sub_pos_counts, summer_sub_neg_counts, winter_sub_pos_counts, winter_sub_neg_counts)
    contingency_submissions = [
        [summer_sub_pos_counts, summer_sub_neg_counts],
        [winter_sub_pos_counts, winter_sub_neg_counts]
    ]

    chi2_submissions = chi2_contingency(contingency_submissions)
    print('Chi-squared p-value submissions (2-seasons, June-September): ', chi2_submissions.pvalue)


    summer_comments_filter = reddit_comments_data.where(functions.col('month').isin(summer_months))
    winter_comments_filter = reddit_comments_data.where(functions.col('month').isin(winter_months))

    summer_comments_neg = summer_comments_filter.where(functions.col('sentiment') == 0)
    summer_comments_pos = summer_comments_filter.where(functions.col('sentiment') == 4)

    winter_comments_neg = winter_comments_filter.where(functions.col('sentiment') == 0)
    winter_comments_pos = winter_comments_filter.where(functions.col('sentiment') == 4)

    summer_com_neg_counts = summer_comments_neg.count()
    summer_com_pos_counts = summer_comments_pos.count()

    winter_com_neg_counts = winter_comments_neg.count()
    winter_com_pos_counts = winter_comments_pos.count()

    print(summer_com_pos_counts, summer_com_neg_counts, winter_com_pos_counts, winter_com_neg_counts)
    contingency_comments = [
        [summer_com_pos_counts, summer_com_neg_counts],
        [winter_com_pos_counts, winter_com_neg_counts]
    ]

    chi2_comments = chi2_contingency(contingency_comments)
    print('Chi-squared p-value comments (2-seasons, June-September): ', chi2_comments.pvalue)


    # Plot of actual vs. expected counts of positive submissions (Summer vs. Non-summer)
    
    season_names = ['Summer', 'Non-summer']
    
    plt.figure(figsize=(14, 12))
    plt.subplot(2, 1, 1)
    
    temporary_x_ticks = np.array([0, 1])
    
    subs_pos_actual = [summer_sub_pos_counts, winter_sub_pos_counts]
    
    subs_pos_expected = chi2_submissions.expected_freq[:,0]
    
    plt.bar(temporary_x_ticks - 0.2, subs_pos_actual, width=0.4, label='Pos Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_pos_expected, width=0.4, label='Pos Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Positive Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Submissions')
    plt.legend()
    
    
    # Plot of actual vs. expected counts of negative submissions (Summer vs. Non-summer)
    
    plt.subplot(2, 1, 2)
    
    subs_neg_actual = [summer_sub_neg_counts, winter_sub_neg_counts]
    
    subs_neg_expected = chi2_submissions.expected_freq[:,1]
    
    plt.bar(temporary_x_ticks - 0.2, subs_neg_actual, width=0.4, label='Neg Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_neg_expected, width=0.4, label='Neg Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Negative Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Submissions')
    plt.legend()
    
    plt.savefig(output + '/actual_vs_expected_subs_2_seasons_june_september.png')
    plt.close()
    
    
    # Plot of actual vs. expected counts of positive comments (Summer vs. Non-summer)

    plt.figure(figsize=(14, 12))
    plt.subplot(2, 1, 1)

    coms_pos_actual = [summer_com_pos_counts, winter_com_pos_counts]

    coms_pos_expected = chi2_comments.expected_freq[:,0]

    plt.bar(temporary_x_ticks - 0.2, coms_pos_actual, width=0.4, label='Pos Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_pos_expected, width=0.4, label='Pos Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Positive Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Comments')
    plt.legend()


    # Plot of actual vs. expected counts of negative comments (Summer vs. Non-summer)

    plt.subplot(2, 1, 2)

    coms_neg_actual = [summer_com_neg_counts, winter_com_neg_counts]

    coms_neg_expected = chi2_comments.expected_freq[:,1]

    plt.bar(temporary_x_ticks - 0.2, coms_neg_actual, width=0.4, label='Neg Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_neg_expected, width=0.4, label='Neg Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Negative Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Comments')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output + '/actual_vs_expected_coms_2_seasons_june_september.png')
    plt.close()
    
    
    # Chi-squared Test (Summer (June-August) vs. Non-summer)
    
    summer_months = [6, 7, 8]
    winter_months = [1, 2, 3, 4, 5, 9, 10, 11, 12]
    
    summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))

    summer_submissions_neg = summer_submissions_filter.where(functions.col('sentiment') == 0)
    summer_submissions_pos = summer_submissions_filter.where(functions.col('sentiment') == 4)

    winter_submissions_neg = winter_submissions_filter.where(functions.col('sentiment') == 0)
    winter_submissions_pos = winter_submissions_filter.where(functions.col('sentiment') == 4)

    summer_sub_neg_counts = summer_submissions_neg.count()
    summer_sub_pos_counts = summer_submissions_pos.count()

    winter_sub_neg_counts = winter_submissions_neg.count()
    winter_sub_pos_counts = winter_submissions_pos.count()

    print(summer_sub_pos_counts, summer_sub_neg_counts, winter_sub_pos_counts, winter_sub_neg_counts)
    contingency_submissions = [
        [summer_sub_pos_counts, summer_sub_neg_counts],
        [winter_sub_pos_counts, winter_sub_neg_counts]
    ]

    chi2_submissions = chi2_contingency(contingency_submissions)
    print('Chi-squared p-value submissions (2-seasons, June-August): ', chi2_submissions.pvalue)


    summer_comments_filter = reddit_comments_data.where(functions.col('month').isin(summer_months))
    winter_comments_filter = reddit_comments_data.where(functions.col('month').isin(winter_months))

    summer_comments_neg = summer_comments_filter.where(functions.col('sentiment') == 0)
    summer_comments_pos = summer_comments_filter.where(functions.col('sentiment') == 4)

    winter_comments_neg = winter_comments_filter.where(functions.col('sentiment') == 0)
    winter_comments_pos = winter_comments_filter.where(functions.col('sentiment') == 4)

    summer_com_neg_counts = summer_comments_neg.count()
    summer_com_pos_counts = summer_comments_pos.count()

    winter_com_neg_counts = winter_comments_neg.count()
    winter_com_pos_counts = winter_comments_pos.count()

    print(summer_com_pos_counts, summer_com_neg_counts, winter_com_pos_counts, winter_com_neg_counts)
    contingency_comments = [
        [summer_com_pos_counts, summer_com_neg_counts],
        [winter_com_pos_counts, winter_com_neg_counts]
    ]

    chi2_comments = chi2_contingency(contingency_comments)
    print('Chi-squared p-value comments (2-seasons, June-August): ', chi2_comments.pvalue)
    
    
    # Plot of actual vs. expected counts of positive submissions (Summer vs. Non-summer)
    
    season_names = ['Summer', 'Non-summer']
    
    plt.figure(figsize=(14, 12))
    plt.subplot(2, 1, 1)
    
    temporary_x_ticks = np.array([0, 1])
    
    subs_pos_actual = [summer_sub_pos_counts, winter_sub_pos_counts]
    
    subs_pos_expected = chi2_submissions.expected_freq[:,0]
    
    plt.bar(temporary_x_ticks - 0.2, subs_pos_actual, width=0.4, label='Pos Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_pos_expected, width=0.4, label='Pos Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Positive Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Submissions')
    plt.legend()
    
    
    # Plot of actual vs. expected counts of negative submissions (Summer vs. Non-summer)
    
    plt.subplot(2, 1, 2)
    
    subs_neg_actual = [summer_sub_neg_counts, winter_sub_neg_counts]
    
    subs_neg_expected = chi2_submissions.expected_freq[:,1]
    
    plt.bar(temporary_x_ticks - 0.2, subs_neg_actual, width=0.4, label='Neg Submissions Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, subs_neg_expected, width=0.4, label='Neg Submissions Expected Count', align='center')
    
    plt.xlabel('Season')
    plt.ylabel('Negative Submissions Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Submissions')
    plt.legend()
    
    plt.savefig(output + '/actual_vs_expected_subs_2_seasons_june_august.png')
    plt.close()
    
    
    # Plot of actual vs. expected counts of positive comments (Summer vs. Non-summer)

    plt.figure(figsize=(14, 12))
    plt.subplot(2, 1, 1)

    coms_pos_actual = [summer_com_pos_counts, winter_com_pos_counts]

    coms_pos_expected = chi2_comments.expected_freq[:,0]

    plt.bar(temporary_x_ticks - 0.2, coms_pos_actual, width=0.4, label='Pos Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_pos_expected, width=0.4, label='Pos Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Positive Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Positive Comments')
    plt.legend()


    # Plot of actual vs. expected counts of negative comments (Summer vs. Non-summer)

    plt.subplot(2, 1, 2)

    coms_neg_actual = [summer_com_neg_counts, winter_com_neg_counts]

    coms_neg_expected = chi2_comments.expected_freq[:,1]

    plt.bar(temporary_x_ticks - 0.2, coms_neg_actual, width=0.4, label='Neg Comments Actual Count', align='center')
    plt.bar(temporary_x_ticks + 0.2, coms_neg_expected, width=0.4, label='Neg Comments Expected Count', align='center')

    plt.xlabel('Season')
    plt.ylabel('Negative Comments Count')
    plt.xticks(temporary_x_ticks, season_names)
    plt.title('Actual vs. Expected Counts for Negative Comments')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output + '/actual_vs_expected_coms_2_seasons_june_august.png')
    plt.close()
    
    
    
if __name__ == "__main__":
    input_submissions = sys.argv[1]
    input_comments = sys.argv[2]
    output = sys.argv[3]
    
    current_directory = os.getcwd()
    subdirectory_path = os.path.join(current_directory, output)
    os.makedirs(subdirectory_path, exist_ok=True)
    
    main(input_submissions, input_comments, output)

#spark-submit reddit_stats.py output-2021-2023-submissions output-2021-2023-comments stats_outputs