# This file is from the data processor (you)
import pandas as pd
from contextlib import redirect_stdout

# Load the dataset (replace with your actual file path)
ad_prediction = pd.read_csv('./input/ad-click-prediction-dataset/ad_click_dataset.csv')

with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        # Basic DataFrame Information
        print("DataFrame Info:")
        ad_prediction.info()  # info() prints directly to stdout
        print("\nFirst 5 rows:")
        print(ad_prediction.head())

        # Distribution of the target variable 'click'
        print("\nDistribution of the target variable 'click':")
        print(ad_prediction['click'].value_counts(normalize=True))  # Normalized percentage of clicks and non-clicks

        # ---------------------------
        # Descriptive Statistics: Numerical Variables
        # ---------------------------
        # For the 'age' column:
        print("\nDescriptive statistics for 'age':")
        print(ad_prediction['age'].describe())

        # Compute individual statistics for 'age'
        mean_age = ad_prediction['age'].mean()
        variance_age = ad_prediction['age'].var()
        std_age = ad_prediction['age'].std()
        median_age = ad_prediction['age'].median()

        print("\nAge Statistics:")
        print("Mean:", mean_age)
        print("Variance:", variance_age)
        print("Standard Deviation:", std_age)
        print("Median:", median_age)

        # For the 'click' column:
        print("\nDescriptive statistics for 'click':")
        print(ad_prediction['click'].describe())

        # Compute individual statistics for 'click'
        mean_click = ad_prediction['click'].mean()
        variance_click = ad_prediction['click'].var()
        std_click = ad_prediction['click'].std()
        median_click = ad_prediction['click'].median()

        print("\nClick Statistics:")
        print("Mean:", mean_click)
        print("Variance:", variance_click)
        print("Standard Deviation:", std_click)
        print("Median:", median_click)

        # ---------------------------
        # Analysis on Categorical Variables
        # ---------------------------
        # Check the frequency counts of various categorical variables
        categorical_vars = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
        for col in categorical_vars:
            print(f"\nValue counts for '{col}':")
            print(ad_prediction[col].value_counts())

        # ---------------------------
        # Grouping and Descriptive Statistics
        # ---------------------------
        # Group by 'gender' and compute descriptive statistics for the 'age' column
        age_stats_by_gender = ad_prediction.groupby('gender')['age'].describe()
        print("\nAge statistics by gender (descriptive):")
        print(age_stats_by_gender)

        # Group by 'device_type' and compute descriptive statistics for the 'click' column
        click_stats_by_device = ad_prediction.groupby('device_type')['click'].describe()
        print("\nClick rate statistics by device type (descriptive):")
        print(click_stats_by_device)

        # Group by 'ad_position' and compute descriptive statistics for the 'click' column
        click_stats_by_ad_position = ad_prediction.groupby('ad_position')['click'].describe()
        print("\nClick rate statistics by ad position (descriptive):")
        print(click_stats_by_ad_position)

        # Group by 'browsing_history' and compute descriptive statistics for the 'click' column
        click_stats_by_browser_history = ad_prediction.groupby('browsing_history')['click'].describe()
        print("\nClick rate statistics by browsing history (descriptive):")
        print(click_stats_by_browser_history)

        # Group by 'time_of_day' and compute descriptive statistics for the 'click' column
        click_stats_by_time_of_day = ad_prediction.groupby('time_of_day')['click'].describe()
        print("\nClick rate statistics by time of day (descriptive):")
        print(click_stats_by_time_of_day)

        # Group by 'device_type', 'ad_position', 'time_of_day' and compute descriptive statistics for the 'click' column
        click_stats_by_multiple_vars = ad_prediction.groupby(['device_type', 'ad_position', 'time_of_day'])['click'].describe()
        print("\nClick rate statistics by multiple variables (descriptive):")
        print(click_stats_by_multiple_vars)

        # ---------------------------
        # Grouping and Aggregating Data
        # ---------------------------
        # Grouping by gender for age statistics
        age_stats = ad_prediction.groupby('gender')['age'].agg(['mean', 'median', 'std', 'var'])
        print("\nAge statistics by gender (aggregating):")
        print(age_stats)

        # Grouping by device type for click statistics
        click_stats = ad_prediction.groupby('device_type')['click'].agg(['mean', 'median', 'std', 'var'])
        print("\nClick rate statistics by device type (aggregating):")
        print(click_stats)

        # Grouping by ad position for click statistics
        ad_position_stats = ad_prediction.groupby('ad_position')['click'].agg(['mean', 'median', 'std', 'var'])
        print("\nClick rate statistics by ad position (aggregating):")
        print(ad_position_stats)

        # Grouping by browsing history for click statistics
        browser_stats = ad_prediction.groupby('browsing_history')['click'].agg(['mean', 'median', 'std', 'var'])
        print("\nClick rate statistics by browsing history (aggregating):")
        print(browser_stats)

        # Grouping by time of day for click statistics
        time_stats = ad_prediction.groupby('time_of_day')['click'].agg(['mean', 'median', 'std', 'var'])
        print("\nClick rate statistics by time of day (aggregating):")
        print(time_stats)

        # Grouping by 'device_type', 'ad_position', 'time_of_day' for click statistics
        multi_stats = ad_prediction.groupby(['device_type', 'ad_position', 'time_of_day'])['click'].agg(['mean', 'median', 'std', 'var'])
        print("\nClick rate statistics by multiple variables (aggregating):")
        print(multi_stats)

        # ---------------------------
        # Visualization
        # ---------------------------
