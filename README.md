# Seasonal Sentiment Analysis on Vancouver Subreddit
This repository contains the code and files used for our course project in CMPT 353 at Simon Fraser University.

Our project's objective is to analyze the sentiments of Vancouver subreddit and its comments by applying statistical models and machine learning. We use statistical model, specifically the chi2_contingency from stats.scipy library to calculate the p-value in order to test our hypotheses.


# Requirements to run our program

- Git
- Python version 3.10 or higher
- **libraries**:
    - sys
    - os 
    - Sparks version 3.5 or higher

Other libraries that need to be installed:
```
pip install --user scipy pandas numpy scikit-learn seaborn matplotlib datasets contractions
```

# Run the program

First we can clone this repository to our directory.

Please cd into the cmpt353project directory that was just created after cloning before running the commands below.

There are **3 python scripts** that need to be executed to run our entire program.
Below is the order they should be run (along with the command to run each of them, and the files expected and produced by each):
        
   - reddit_data_cleaning.py (data cleaning script)
 
        command to run the script:
        
        ```
        spark-submit reddit_data_cleaning.py reddit-subset-2021-2023
        ```
        
        files expected: a reddit-subset-2021-2023 directory
        
        files produced: a reddit-subset-2021-2023-submissions directory, and a reddit-subset-2021-2023-comments directory
        
        
   - reddit_data_sentiment_analysis.py (sentiment analysis script)
   
        command to run the script:
        
        ```
        spark-submit reddit_data_sentiment_analysis.py reddit-subset-2021-2023-submissions reddit-subset-2021-2023-comments output-2021-2023
        ```
        
        files expected: a reddit-subset-2021-2023-submissions directory, and a reddit-subset-2021-2023-comments directory
        
        files produced: an output-2021-2023-submissions directory and an output-2021-2023-comments directory 
        
        
   - reddit_stats.py (statistical analysis script)
   
        command to run the script:
        
        ```
        spark-submit reddit_stats.py output-2021-2023-submissions output-2021-2023-comments stats-output-2021-2023
        ```
        
        files expected: an output-2021-2023-submissions directory and an output-2021-2023-comments directory (both contain sentiment analysis output)
        
        files produced: a stats-output-2021-2023 directory (containing the plots produced) 
        


    





















