# Contains common routines and variable definitions
# Prevents overloading the exploration.ipynb notebook

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import PyPDF2

############################################################
#                                                          #
#        ------  Data Understanding     -------            #
#                                                          #
############################################################ 

def extract_stop_codes(pdf_path):
    """
    Extract stop codes from a PDF document.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple: Lists of primary care codes, mental health codes, and all other specialties codes.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        primary_care_codes = []
        mental_health_codes = []
        all_other_specialties_codes = []
        
        # Read text from pages 2 to 4
        for page_num in range(1, 4):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # Extract stop codes based on section headers
            if 'Primary Care' in text:
                primary_care_section = text.split('Primary Care')[1].split('Mental Health')[0]
                primary_care_codes += [line.split()[0] for line in primary_care_section.split('\n') if line.strip()]
            
            if 'Mental Health' in text:
                mental_health_section = text.split('Mental Health')[1].split('All Other Specialties')[0]
                mental_health_codes += [line.split()[0] for line in mental_health_section.split('\n') if line.strip()]
            
            if 'All Other Specialties' in text:
                all_other_specialties_section = text.split('All Other Specialties')[1]
                all_other_specialties_codes += [line.split()[0] for line in all_other_specialties_section.split('\n') if line.strip()]
        
        return primary_care_codes, mental_health_codes, all_other_specialties_codes

def plot_histograms_numeric(data):
    """
    Plot histograms for specified numeric variables in the DataFrame.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle('Histograms of Key Numeric Variables', fontsize=16)
    
    variables = ['dta', 'dts', 'dtc']
    bins_settings = [50, 50, 50]  
    xlims = [(0, 100), (0, 250), (0, 500)] 
    
    for ax, var, bins, xlim in zip(axes, variables, bins_settings, xlims):
        data[var].hist(bins=bins, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {var}', fontsize=14)
        ax.set_xlabel(f'Values of {var}')
        ax.set_xlim(xlim)

def plot_top_category_counts(data):
    """
    Plot count plots for categorical variables, focusing on the top categories.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
    """
    categorical_vars = data.select_dtypes(include=['category']).columns.tolist()
    plt.figure(figsize=(10, 5 * len(categorical_vars)))

    for i, var in enumerate(categorical_vars):
        counts = data[var].value_counts()
        top_categories = counts.index[:10]
        top_data = data[data[var].isin(top_categories)]

        plt.subplot(len(categorical_vars), 1, i + 1)
        palette = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=10, reverse=True, dark=0.4, light=0.95)
        sns.countplot(x=var, data=top_data, palette=palette, order=top_categories)
        plt.title(f'Count Plot of {var}')
        plt.xlabel(f'{var} Categories')  
        plt.ylabel('Frequency')  
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
        
def analyze_working_patterns(data):
    """
    Analyze and plot working patterns by hour and weekday.

    Args:
        data (DataFrame): The pandas DataFrame containing 'activitydatetime'.
    """
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    weekday_counts = data['weekday'].value_counts().sort_index()
    hour_counts = data['hour'].value_counts().sort_index()
    
    weekday_norm = (weekday_counts - weekday_counts.min()) / (weekday_counts.max() - weekday_counts.min())
    hour_norm = (hour_counts - hour_counts.min()) / (hour_counts.max() - hour_counts.min())
    
    blue_palette_weekday = sns.color_palette("Blues", as_cmap=True)(weekday_norm)
    blue_palette_hour = sns.color_palette("Blues", as_cmap=True)(hour_norm)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette=blue_palette_weekday)
    plt.title('Number of Patients by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette=blue_palette_hour)
    plt.title('Number of Patients by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 24))
    plt.show()
    
    data.drop(["hour", "weekday"], axis=1, inplace=True)

def filter_dates_after(data, datetime_col, cutoff_date):
    """
    Filter DataFrame to keep rows where datetime is after the cutoff date.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        datetime_col (str): Name of the datetime column.
        cutoff_date (str): The cutoff date in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        DataFrame: Filtered DataFrame.
    """
    if data[datetime_col].dtype == 'object':
        data[datetime_col] = pd.to_datetime(data[datetime_col])

    cutoff_datetime = pd.to_datetime(cutoff_date)
    filtered_data = data[data[datetime_col] > cutoff_datetime]

    return filtered_data

# Function to filter stop codes
def filter_stop_codes(df, primary_care_codes, mental_health_codes, all_other_specialties_codes):
    valid_codes = set(map(int, primary_care_codes + mental_health_codes + all_other_specialties_codes))
    return df[df['stopcode'].isin(valid_codes)]

# Function to add binary variable "type_soins"
def add_type_soins(df, primary_care_codes):
    primary_care_codes = set(map(int, primary_care_codes))
    df['typecare'] = df['stopcode'].apply(lambda x: 0 if x in primary_care_codes else 1)
    return df

############################################################
#                                                          #
#        ------    Data Preparation     -------            #
#                                                          #
############################################################ 

def display_missing_values(data):
    """
    Display count and percentage of missing values per column.

    Args:
        data (DataFrame): The pandas DataFrame to analyze.

    Returns:
        Series: Percentages of missing values per column.
    """
    missing_values = data.isnull().sum()
    additional_missing = (data['state'] == '*Missing*').sum()
    missing_values['state'] += additional_missing

    total_rows = data.shape[0]
    missing_percentages = (missing_values / total_rows) * 100
    print("Percentage of missing values per column:")
    return missing_percentages

def detect_outliers_by_IQR(data, feature):
    """
    Identify outliers in a feature using the IQR method.

    Args:
        data (DataFrame): The DataFrame containing the data.
        feature (str): The column to check for outliers.

    Returns:
        Index: Indices of outliers.
    """
    Q1 = np.percentile(data[feature], 25)
    Q3 = np.percentile(data[feature], 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outliers = data[(data[feature] < Q1 - outlier_step) | (data[feature] > Q3 + outlier_step)].index
    return outliers

def visualize_outliers(data):
    """
    Visualize the count of outliers in each numeric column.

    Args:
        data (DataFrame): The DataFrame to analyze.
    """
    outlier_indices_dict = {}

    for column in data.select_dtypes(include=[np.number]).columns:
        outlier_indices = detect_outliers_by_IQR(data, column)
        outlier_indices_dict[column] = len(outlier_indices)

    print("Outliers (IQR) Count Per Numeric Column:")
    print(outlier_indices_dict)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(outlier_indices_dict.keys()), y=list(outlier_indices_dict.values()), palette='viridis')
    plt.title('Outlier Counts in Numeric Columns')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.show()        
        

def remove_outliers_from_dataframe(data):
    """
    Remove outliers from all numeric columns using the IQR method.

    Args:
        data (DataFrame): The pandas DataFrame to clean.

    Returns:
        int: Total number of outliers removed.
    """
    outlier_indices = set()
    for col in data.select_dtypes(include=[np.number]).columns:
        outlier_indices.update(detect_outliers_by_IQR(data, col))

    len_outliers = len(outlier_indices)
    data.drop(index=list(outlier_indices), inplace=True)

    print(f"Data shape after outlier removal: {data.shape}")
    print(f"Total outliers removed: {len_outliers}")

    return len_outliers

def plot_consultation_counts(data):
    """
    Plot the count of consultations for each category in 'NoShow'.

    Args:
        data (DataFrame): The pandas DataFrame containing the 'NoShow' column.
    """
    plt.figure(figsize=(4, 4))
    count_series = data['NoShow'].value_counts()
    plt.pie(count_series, labels=count_series.index, autopct='%1.2f%%', startangle=5, textprops={'fontsize':8})
    plt.axis('equal')
    plt.title('Percentage of Consultations per Category')
    plt.show()  

def filter_categories(df, categories):
    """
    Filter the DataFrame to keep rows with 'NoShow' in specified categories.

    Args:
        df (DataFrame): The pandas DataFrame to filter.
        categories (list): Categories to keep.

    Returns:
        DataFrame: Filtered DataFrame.
    """
    df = df[df['NoShow'].isin(categories)]
    return df

def remove_outliers_from_columns(data, columns):
    """
    Remove outliers from specified columns using the IQR method.

    Args:
        data (DataFrame): The pandas DataFrame to clean.
        columns (list): List of column names.

    Returns:
        dict: Counts of outliers removed per column.
    """
    total_outliers_removed = 0
    outliers_removed_per_column = {}

    for column in columns:
        outlier_indices = detect_outliers_by_IQR(data, column)
        len_outliers = len(outlier_indices)
        data.drop(index=outlier_indices, inplace=True)
        outliers_removed_per_column[column] = len_outliers
        total_outliers_removed += len_outliers

        print(f"Data shape after outlier removal in column {column}: {data.shape}")
        print(f"Total outliers removed from column {column}: {len_outliers}")

    print(f"Total outliers removed from all columns: {total_outliers_removed}")
    return outliers_removed_per_column

def plot_no_show_comparaison(data):
    """
    Plot the number of no-show patients in Community Care and VHA.

    Args:
        data (DataFrame): The pandas DataFrame containing the consultation data.
    """
    community_care_no_shows = data[(data['NoShow'] == 1) & (data['non_va'] == 1)].shape[0]
    vha_no_shows = data[(data['NoShow'] == 1) & (data['non_va'] == 0)].shape[0]

    categories = ['Community Care No-Shows', 'VHA No-Shows']
    values = [community_care_no_shows, vha_no_shows]
    colors = ['#add8e6', '#87ceeb'] 

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors)
    plt.xlabel('Categories')
    plt.ylabel('Number of No-Shows')
    plt.title('Comparison of No-Shows: Community Care vs VHA')
    plt.show()

def plot_patient_flow(data, datetime_col='activitydatetime', disp_col='NoShow'):
    """
    Plot patient flow based on hour, day of week, and day of month.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        datetime_col (str): Name of the datetime column.
        disp_col (str): Name of the column indicating show or no-show.
    """
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['day_of_week_name'] = data['day_of_week'].map(days)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='hour', y=disp_col, estimator='mean', ci=None)
    plt.title('Average No-Show-up Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(range(0, 24))  
    plt.show()
    data.sort_values('day_of_week', inplace=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='day_of_week_name', y=disp_col, estimator='mean', ci=None)
    plt.title('Average No-Show-up Rate by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(rotation=45)  
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='day_of_month', y=disp_col, estimator='mean', ci=None)
    plt.title('Average No-Show-up Rate by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(range(1, 32))  
    plt.show()
    data.drop('day_of_week_name', axis=1, inplace=True)

def separate_stopcodes_by_no_show(data):
    """
    Separate data into stopcodes with and without no-shows.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.

    Returns:
        tuple: DataFrames and list for stopcodes with and without no-shows.
    """
    no_shows = data[data['NoShow'] == 1]
    stopcodes_with_no_shows = no_shows['stopcode'].unique()
    data_with_no_shows = data[data['stopcode'].isin(stopcodes_with_no_shows)]
    data_without_no_shows = data[~data['stopcode'].isin(stopcodes_with_no_shows)]
    stopcodes_without_no_shows = data_without_no_shows['stopcode'].unique()
    
    return data_with_no_shows, data_without_no_shows, stopcodes_without_no_shows
        
def count_no_shows_by_establishment(data):
    """
    Count no-shows per establishment and sort descending.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.

    Returns:
        Series: Counts of no-shows per establishment.
    """
    no_shows = data[data['NoShow'] == 1]
    no_shows_by_establishment = no_shows['sta3n'].value_counts(ascending=False)
    return no_shows_by_establishment 

def add_temporal_features(data, datetime_column):
    """
    Add temporal features from a datetime column.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        datetime_column (str): Name of the datetime column.

    Returns:
        DataFrame: DataFrame with new temporal features.
    """
    data[datetime_column] = pd.to_datetime(data[datetime_column])
    data['day_of_week'] = data[datetime_column].dt.dayofweek.astype('uint8')
    data['day_of_month'] = data[datetime_column].dt.day.astype('uint8')
    data['month'] = data[datetime_column].dt.month.astype('uint8')
    data['hour'] = data[datetime_column].dt.hour.astype('uint8')
    return data

def decompose_disp_column(data, disp_column="disp"):
    """
    Decompose 'disp' column into 'discontinued' and 'NoShow' columns.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        disp_column (str): Name of the 'disp' column to decompose.

    Returns:
        DataFrame: DataFrame with new columns added.
    """
    data['NoShow'] = data[disp_column].map({'DISCONTINUED': 0, 'CANCELLED': 1, 'COMPLETE/UPDATE': 0}).astype('uint8')
    data['discontinued'] = data[disp_column].map({'DISCONTINUED': 1, 'CANCELLED': 0, 'COMPLETE/UPDATE': 0}).astype('uint8')
    data.drop('disp', axis=1, inplace=True)
    return data

def plot_patient_flow_(data, disp_col='NoShow'):
    """
    Plot patient flow based on hour, day of week, and day of month with inverted y-axis.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        disp_col (str): Name of the column indicating show or no-show.
    """
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['day_of_week_name'] = data['day_of_week'].map(days)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='hour', y=disp_col, estimator='mean', ci=None)
    plt.gca().invert_yaxis()
    plt.title('Average No-Show-up Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(range(0, 24))  
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='day_of_week_name', y=disp_col, estimator='mean', ci=None)
    plt.gca().invert_yaxis()
    plt.title('Average No-Show-up Rate by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(rotation=45)  
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='day_of_month', y=disp_col, estimator='mean', ci=None)
    plt.gca().invert_yaxis()
    plt.title('Average No-Show-up Rate by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.ylabel('Average No-Show-up Rate')
    plt.xticks(range(1, 32))  
    plt.show()

def plot_patient_flow_segmented(data, disp_col='NoShow', estimator=np.mean):
    """
    Plot patient flow segmented into four 7-day periods of the month.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.
        disp_col (str): Column indicating show or no-show.
        estimator (function): Statistical function to estimate the average.
    """
    plt.figure(figsize=(12, 6))

    data['week_segment'] = data['day_of_month'].apply(lambda x: (x - 1) // 7 + 1)
    data['day_in_week_segment'] = data['day_of_month'].apply(lambda x: ((x - 1) % 7) + 1)

    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for week in sorted(data['week_segment'].unique()):
        subset = data[data['week_segment'] == week]
        sns.lineplot(x='day_in_week_segment', y=disp_col, data=subset, estimator=estimator, ci=None,
                     label=f'Week {week}')

    plt.title('Average No-Show-up Rate by Day of the Week Across Different Weeks')
    plt.xlabel('Day of the Week')
    plt.ylabel('Estimated No-Show-up Rate')
    plt.xticks(range(1, 8), day_labels, rotation=45)
    plt.legend(title='Week Segment')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_no_show_rate_by_typecare(data):
    """
    Plot no-show rate based on type of care (Primary vs Specialized).

    Args:
        data (DataFrame): The pandas DataFrame containing the consultation data.
    """
    primary_care_total = data[data['typecare'] == 0].shape[0]
    primary_care_no_shows = data[(data['NoShow'] == 1) & (data['typecare'] == 0)].shape[0]
    primary_care_no_show_rate = (primary_care_no_shows / primary_care_total) * 100 if primary_care_total > 0 else 0

    specialized_care_total = data[data['typecare'] == 1].shape[0]
    specialized_care_no_shows = data[(data['NoShow'] == 1) & (data['typecare'] == 1)].shape[0]
    specialized_care_no_show_rate = (specialized_care_no_shows / specialized_care_total) * 100 if specialized_care_total > 0 else 0

    categories = ['Primary Care No-Show Rate', 'Specialized Care No-Show Rate']
    values = [primary_care_no_show_rate, specialized_care_no_show_rate]
    colors = ['#4682b4', '#add8e6'] 

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors)
    plt.ylabel('No-Show Rate (%)')
    plt.show()

def plot_service_time_by_hour_and_weekday_vha(data):
    """
    Plot average service time by hour and weekday for VHA and community care.

    Args:
        data (DataFrame): The pandas DataFrame containing 'activitydatetime', 'service_time', and 'non_va' columns.
    """
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    vha_data = data[data['non_va'] == 0]
    community_data = data[data['non_va'] == 1]
    
    overall_hourly_avg = data.groupby('hour')['service_time'].mean()
    overall_weekly_avg = data.groupby('weekday')['service_time'].mean()
    vha_hourly_avg = vha_data.groupby('hour')['service_time'].mean()
    community_hourly_avg = community_data.groupby('hour')['service_time'].mean()
    vha_weekly_avg = vha_data.groupby('weekday')['service_time'].mean()
    community_weekly_avg = community_data.groupby('weekday')['service_time'].mean()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    
    axes[0].plot(overall_hourly_avg.index, overall_hourly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[0].plot(vha_hourly_avg.index, vha_hourly_avg.values, label='VHA', marker='o', linestyle='-', color='cyan')
    axes[0].plot(community_hourly_avg.index, community_hourly_avg.values, label='Community Care', marker='o', linestyle='-', color='blue')
    axes[0].set_xlabel('Hour of the Day')
    axes[0].set_ylabel('Average Service Time')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(range(0, 24)) 
    
    axes[1].plot(overall_weekly_avg.index, overall_weekly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[1].plot(vha_weekly_avg.index, vha_weekly_avg.values, label='VHA', marker='o', linestyle='-', color='cyan')
    axes[1].plot(community_weekly_avg.index, community_weekly_avg.values, label='Community Care', marker='o', linestyle='-', color='blue')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Average Service Time')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(range(0, 7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.tight_layout()
    plt.show()
    
    data.drop(["hour", "weekday"], axis=1, inplace=True)
        
def plot_service_time_by_hour_and_weekday_type_care(data):
    """
    Plot average service time by hour and weekday for primary and specialized care.

    Args:
        data (DataFrame): The pandas DataFrame containing 'activitydatetime', 'service_time', and 'typecare' columns.
    """
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    primary_care_data = data[data['typecare'] == 0]
    specialized_care_data = data[data['typecare'] == 1]
    
    overall_hourly_avg = data.groupby('hour')['service_time'].mean()
    overall_weekly_avg = data.groupby('weekday')['service_time'].mean()
    primary_care_hourly_avg = primary_care_data.groupby('hour')['service_time'].mean()
    specialized_care_hourly_avg = specialized_care_data.groupby('hour')['service_time'].mean()
    primary_care_weekly_avg = primary_care_data.groupby('weekday')['service_time'].mean()
    specialized_care_weekly_avg = specialized_care_data.groupby('weekday')['service_time'].mean()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    
    axes[0].plot(overall_hourly_avg.index, overall_hourly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[0].plot(primary_care_hourly_avg.index, primary_care_hourly_avg.values, label='Primary Care', marker='o', linestyle='-', color='blue')
    axes[0].plot(specialized_care_hourly_avg.index, specialized_care_hourly_avg.values, label='Specialized Care', marker='o', linestyle='-', color='cyan')
    axes[0].set_xlabel('Hour of the Day')
    axes[0].set_ylabel('Average Service Time')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(range(0, 24)) 
    
    axes[1].plot(overall_weekly_avg.index, overall_weekly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[1].plot(primary_care_weekly_avg.index, primary_care_weekly_avg.values, label='Primary Care', marker='o', linestyle='-', color='blue')
    axes[1].plot(specialized_care_weekly_avg.index, specialized_care_weekly_avg.values, label='Specialized Care', marker='o', linestyle='-', color='cyan')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Average Service Time')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(range(0, 7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.tight_layout()
    plt.show()
    
    data.drop(["hour", "weekday"], axis=1, inplace=True)

def count_consultations(data):
    """
    Count the number of consultations in Community Care and VHA.

    Args:
        data (DataFrame): The pandas DataFrame containing the consultation data.

    Returns:
        dict: Counts of consultations.
    """
    community_care_count = data[data['non_va'] == 1].shape[0]
    vha_count = data[data['non_va'] == 0].shape[0]

    result = {
        'Community Care (CC) Consultations': community_care_count,
        'VHA Consultations': vha_count
    }

    return result

def plot_no_show_rate_comparaison(data):
    """
    Plot the no-show rate for patients in Community Care vs VHA.

    Args:
        data (DataFrame): The pandas DataFrame containing the consultation data.
    """
    community_care_total = data[data['non_va'] == 1].shape[0]
    community_care_no_shows = data[(data['NoShow'] == 1) & (data['non_va'] == 1)].shape[0]
    community_care_no_show_rate = (community_care_no_shows / community_care_total) * 100 if community_care_total > 0 else 0

    vha_total = data[data['non_va'] == 0].shape[0]
    vha_no_shows = data[(data['NoShow'] == 1) & (data['non_va'] == 0)].shape[0]
    vha_no_show_rate = (vha_no_shows / vha_total) * 100 if vha_total > 0 else 0

    categories = ['Community Care No-Show Rate', 'VHA No-Show Rate']
    values = [community_care_no_show_rate, vha_no_show_rate]
    colors = ['#4682b4','#add8e6'] 

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors)
    plt.ylabel('No-Show Rate (%)')
    plt.show()

def separate_stopcodes_by_no_show_(data):
    """
    Separate data into stopcodes with and without no-shows, including list of stopcodes without no-shows.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.

    Returns:
        tuple: DataFrames and list for stopcodes with and without no-shows.
    """
    no_shows = data[data['NoShow'] == 1]
    stopcodes_with_no_shows = no_shows['stopcode'].unique()
    data_with_no_shows = data[data['stopcode'].isin(stopcodes_with_no_shows)]
    data_without_no_shows = data[~data['stopcode'].isin(stopcodes_with_no_shows)]
    stopcodes_without_no_shows = data_without_no_shows['stopcode'].unique()
    
    return data_with_no_shows, data_without_no_shows, stopcodes_without_no_shows

def standardize_numeric_columns(data, exclude_column='patientsid'):
    """
    Standardize numeric columns except the excluded column.

    Args:
        data (DataFrame): The pandas DataFrame to standardize.
        exclude_column (str): Column to exclude from standardization.

    Returns:
        tuple: Standardized DataFrame and scaler object.
    """
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if exclude_column in numeric_cols:
        numeric_cols.remove(exclude_column)
    
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data, scaler

def encode_categorical_columns(data):
    """
    Encode categorical columns in the DataFrame.

    Args:
        data (DataFrame): The pandas DataFrame containing the data.

    Returns:
        DataFrame: DataFrame with encoded categorical columns.
    """
    label_encoder = LabelEncoder()
    
    for column in data.select_dtypes(include=['category']).columns:
        is_numeric = all(str(cat).isdigit() for cat in data[column].cat.categories)

        if not is_numeric:
            data[column] = label_encoder.fit_transform(data[column])
        else:
            data[column] = data[column].astype(int)
            
    return data

def analyze_working_patterns_(data):
    """
    Analyze and plot working patterns by hour and weekday.

    Args:
        data (DataFrame): The pandas DataFrame containing 'activitydatetime'.
    """
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    weekday_counts = data['weekday'].value_counts().sort_index()
    hour_counts = data['hour'].value_counts().sort_index()
    
    weekday_norm = (weekday_counts - weekday_counts.min()) / (weekday_counts.max() - weekday_counts.min())
    hour_norm = (hour_counts - hour_counts.min()) / (hour_counts.max() - hour_counts.min())
    
    blue_palette_weekday = sns.color_palette("Blues", as_cmap=True)(weekday_norm)
    blue_palette_hour = sns.color_palette("Blues", as_cmap=True)(hour_norm)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette=blue_palette_weekday)
    plt.title('Number of Patients by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette=blue_palette_hour)
    plt.title('Number of Patients by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 24))
    plt.show()
    
    data.drop(["hour", "weekday"], axis=1, inplace=True)

############################################################
#                                                          #
#        ------      Specific Service Time       ------     #
#                                                          #
############################################################ 

def remove_outliers_from_column(data, column):
    """
    Remove outliers from a specified column using the IQR method.

    Args:
        data (DataFrame): The pandas DataFrame to clean.
        column (str): Name of the column.

    Returns:
        int: Number of outliers removed.
    """
    outlier_indices = detect_outliers_by_IQR(data, column)
    len_outliers = len(outlier_indices)
    data.drop(index=outlier_indices, inplace=True)

    print(f"Data shape after outlier removal in column {column}: {data.shape}")
    print(f"Total outliers removed from column {column}: {len_outliers}")

    return len_outliers

def plot_service_time_by_hour_and_weekday(data):
    """
    Plot average service time by hour and weekday, differentiating between VHA and community care.

    Args:
        data (DataFrame): The pandas DataFrame containing 'activitydatetime', 'ts', and 'non_va' columns.
    """
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    vha_data = data[data['non_va'] == '0']
    community_data = data[data['non_va'] == '1']
    
    overall_hourly_avg = data.groupby('hour')['ts'].mean()
    overall_weekly_avg = data.groupby('weekday')['ts'].mean()
    vha_hourly_avg = vha_data.groupby('hour')['ts'].mean()
    community_hourly_avg = community_data.groupby('hour')['ts'].mean()
    vha_weekly_avg = vha_data.groupby('weekday')['ts'].mean()
    community_weekly_avg = community_data.groupby('weekday')['ts'].mean()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    
    axes[0].plot(overall_hourly_avg.index, overall_hourly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[0].plot(vha_hourly_avg.index, vha_hourly_avg.values, label='VHA', marker='o', linestyle='-', color='blue')
    axes[0].plot(community_hourly_avg.index, community_hourly_avg.values, label='Community Care', marker='o', linestyle='-', color='green')
    axes[0].set_title('Average Service Time by Hour of Day')
    axes[0].set_xlabel('Hour of the Day')
    axes[0].set_ylabel('Average Service Time (ts)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(range(0, 24))
    
    axes[1].plot(overall_weekly_avg.index, overall_weekly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[1].plot(vha_weekly_avg.index, vha_weekly_avg.values, label='VHA', marker='o', linestyle='-', color='blue')
    axes[1].plot(community_weekly_avg.index, community_weekly_avg.values, label='Community Care', marker='o', linestyle='-', color='green')
    axes[1].set_title('Average Service Time by Weekday')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Average Service Time (ts)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(range(0, 7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.tight_layout()
    plt.show()
    
    data.drop(["hour","weekday"], axis=1, inplace=True)
