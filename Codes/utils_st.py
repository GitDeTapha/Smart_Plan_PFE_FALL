"""
utils_st.py
   contient les routines communnes et définitions de variable
   Permet de ne pas surcharger le notebook Prediction_Service_Time.ipynb
"""


# Chargement des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

############################################################
#                                                          #
#        ------  Data Understanding     -------            #
#                                                          #
############################################################ 

def plot_histograms_numeric(data):
    """
    Plots histograms for specified numeric variables within the dataframe.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
    
    This function creates a 4x1 grid of histogram plots for the variables 'dta', 'dts', 'dtc'.
    Each histogram is customized with specific bin settings and x-axis limits to enhance the data visualization.
    """
    # Create a subplot grid of 4 rows by 1 column, with specified figure size and vertical spacing
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle('Histograms of Key Numeric Variables', fontsize=16)
    
    # Lists of variable names, bin counts, and x-axis limits for each plot
    variables = ['dta', 'dts', 'dtc']
    bins_settings = [50, 50, 50]  
    xlims = [(0, 100), (0, 250), (0, 500)] 
    
    # Loop over each variable and the corresponding axes object
    for ax, var, bins, xlim in zip(axes, variables, bins_settings, xlims):
        # Plot histogram 
        data[var].hist(bins=bins, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {var}', fontsize=14)
        ax.set_xlabel(f'Values of {var}')
        ax.set_xlim(xlim)

def plot_top_category_counts(data):
    """
    Plots count plots for categorical variables in the dataframe, focusing only on the top 25 categories
    for readability and memory efficiency.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
    
    This function identifies all categorical variables, calculates their frequency, and plots count plots
    for the top 10 most frequent categories.
    """
    # Identify categorical variables
    categorical_vars = data.select_dtypes(include=['category']).columns.tolist()

    # Setting up the figure to accommodate all subplots
    plt.figure(figsize=(10, 5 * len(categorical_vars)))  # Height adjusted based on number of variables

    # Iterate over each categorical variable
    for i, var in enumerate(categorical_vars):
        # Calculate the frequency of each category
        counts = data[var].value_counts()

        # Keep only the top 25 categories
        top_categories = counts.index[:10]
        top_data = data[data[var].isin(top_categories)]

        # Create a subplot for each categorical variable
        plt.subplot(len(categorical_vars), 1, i + 1)
        
        # Generate a color palette
        palette = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=10, reverse=True, dark=0.4, light=0.95)

        # Plot count plot with sorted categories by count
        sns.countplot(x=var, data=top_data, palette=palette, order=top_categories)
        plt.title(f'Count Plot of {var}')
        plt.xlabel(f'{var} Categories')  
        plt.ylabel('Frequency')  
        plt.xticks(rotation=45)  # Rotate the category labels to make them readable

    # Adjust layout to prevent overlap and ensure everything fits well
    plt.tight_layout()
    plt.show()
    
    
def analyze_working_patterns(data):
    # Ensure 'activitydatetime' is in datetime format
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    
    # Extract hour and weekday from 'activitydatetime'
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    # Group by the weekday to find the number of patients per day of the week
    weekday_counts = data['weekday'].value_counts().sort_index()
    
    # Group by the hour to find the number of patients per hour of the day
    hour_counts = data['hour'].value_counts().sort_index()
    
    # Plotting the number of patients by day of the week
    plt.figure(figsize=(10, 5))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette='viridis')
    plt.title('Number of Patients by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
    
    # Plotting the number of patients by hour of the day
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='viridis')
    plt.title('Number of Patients by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Patients')
    plt.xticks(range(0, 24))
    plt.show()
    
    data.drop(["hour","weekday"], axis=1, inplace=True)
    
    



############################################################
#                                                          #
#        ------    Data Preparation     -------            #
#                                                          #
############################################################ 

def display_missing_values(data):
    """
    Displays the count and percentage of missing values per column in the DataFrame, including specific markers treated as missing,
    and visualizes these counts as percentages.
    
    Parameters:
        data (DataFrame): The pandas DataFrame to analyze for missing values.
    """
    # Calculate the number of NaN directly
    missing_values = data.isnull().sum()
    
    # Add to the count of NaNs, those marked as '*Missing*' for the 'state' column
    additional_missing = (data['state'] == '*Missing*').sum()
    missing_values['state'] += additional_missing

    # Display the raw count of missing values for each column
    print("Missing Values Count Per Column:")
    print(missing_values)

    # Convert counts to percentage of the total number of rows
    total_rows = len(data)
    missing_percentages = (missing_values / total_rows) * 100

    # Filter to keep only columns with missing values
    missing_percentages = missing_percentages[missing_percentages > 0]

    if not missing_percentages.empty:
        print("\nMissing Values Percentage Per Column:")
        print(missing_percentages)

        # Visualize the missing values percentages if there are any
        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_percentages.index, y=missing_percentages.values)
        plt.title('Missing Values Percentage')
        plt.ylabel('Percentage of Total Entries')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.show()
    else:
        print("\nNo missing values found in the DataFrame.")
        
        
def detect_outliers_by_IQR(data, feature):
    """
    Identifies outliers in a specified feature column of a DataFrame using the IQR method.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        feature (str): The column in the DataFrame for which to detect outliers.

    Returns:
        Index: Indices of the outliers in the DataFrame.
    """
    # Calculate the first and third quartile, and the IQR
    Q1 = np.percentile(data[feature], 25)
    Q3 = np.percentile(data[feature], 75)
    IQR = Q3 - Q1

    # Determine outliers using 1.5 times the IQR from Q1 and Q3
    outlier_step = 1.5 * IQR
    outliers = data[(data[feature] < Q1 - outlier_step) | (data[feature] > Q3 + outlier_step)].index
    return outliers

def visualize_outliers(data):
    """
    Applies outlier detection to each numeric column in the DataFrame and visualizes the count of outliers.

    Parameters:
        data (DataFrame): The DataFrame to analyze.
    """
    outlier_indices_dict = {}

    # Apply outlier detection for each numeric column
    for column in data.select_dtypes(include=[np.number]).columns:
        outlier_indices = detect_outliers_by_IQR(data, column)
        outlier_indices_dict[column] = len(outlier_indices)

    # Print and visualize the counts of outliers per column
    print("Outliers (IQR) Count Per Numeric Column:")
    print(outlier_indices_dict)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(outlier_indices_dict.keys()), y=list(outlier_indices_dict.values()), palette='viridis')
    plt.title('Outlier Counts in Numeric Columns')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()        
        

def remove_outliers_from_dataframe(data):
    """
    Removes outliers in-place from all numeric columns of the DataFrame using the Interquartile Range (IQR) method.
    This approach updates the original DataFrame directly, conserving memory by not creating a copy of the DataFrame.

    Parameters:
        data (DataFrame): The pandas DataFrame from which to remove outliers.

    Returns:
        int: The count of total outliers removed.
    """
    # Collect all outlier indices from each numeric column
    outlier_indices = set()
    for col in data.select_dtypes(include=[np.number]).columns:
        outlier_indices.update(detect_outliers_by_IQR(data, col))

    # Count of total outliers detected
    len_outlier_interquartile = len(outlier_indices)

    # Remove outliers and update the DataFrame in place
    data.drop(index=list(outlier_indices), inplace=True)

    print(f"Data shape after outlier removal: {data.shape}")
    print(f"Total outliers removed: {len_outlier_interquartile}")

    return len_outlier_interquartile


def plot_consultation_counts(data):
    """
    Plots the count of consultations for each category in the 'disp' column using a pie chart.
    
    Args:
    data (DataFrame): The pandas DataFrame containing the 'disp' column.
    """
    plt.figure(figsize=(8, 8))
    # Calculate the counts of each category
    count_series = data['disp'].value_counts()
    
    # Create the pie chart with percentages
    plt.pie(count_series, labels=count_series.index, autopct='%1.1f%%', startangle=90)
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')  
    plt.title('Percentage of Consultations per Category')
    plt.show()


    
def filter_categories(df, categories):
    """
    Filters the DataFrame to keep only the rows with 'disp' values in the specified categories.
    Optionally, removes the 'disp' column from the DataFrame.
    
    Args:
    data (DataFrame): The pandas DataFrame to filter.
    categories (list): A list of categories to keep in the 'disp' column.
    
    Returns:
    DataFrame: The filtered DataFrame.
    """
    df = df[df['disp'].isin(categories)]
    return df

def remove_outliers_from_column(data, column):
    """
    Removes outliers in-place from a specified column of the DataFrame using the Interquartile Range (IQR) method.
    This approach updates the original DataFrame directly, conserving memory by not creating a copy of the DataFrame.

    Parameters:
        data (DataFrame): The pandas DataFrame from which to remove outliers.
        column (str): The name of the column from which to remove outliers.

    Returns:
        int: The count of total outliers removed from the specified column.
    """
    # Detect outliers for the specified column
    outlier_indices = detect_outliers_by_IQR(data, column)

    # Count of total outliers detected
    len_outlier_interquartile = len(outlier_indices)

    # Remove outliers and update the DataFrame in place
    data.drop(index=outlier_indices, inplace=True)

    print(f"Data shape after outlier removal in column {column}: {data.shape}")
    print(f"Total outliers removed from column {column}: {len_outlier_interquartile}")

    return len_outlier_interquartile



def plot_service_time_by_hour_and_weekday(data):
    """
    Plots the average service time by hour of day and by weekday, differentiating between VHA (non_va == 0)
    and community care (non_va == 1), and includes the overall average, displaying the plots side by side for comparative analysis.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the 'activitydatetime', 'ts', and 'non_va' columns.
    """
    # Ensure 'activitydatetime' is in datetime format and extract hour and weekday
    data['activitydatetime'] = pd.to_datetime(data['activitydatetime'])
    data['hour'] = data['activitydatetime'].dt.hour
    data['weekday'] = data['activitydatetime'].dt.weekday
    
    # Calculate the mean service time for VHA, community care, and overall
    vha_data = data[data['non_va'] == '0']
    community_data = data[data['non_va'] == '1']
    
    overall_hourly_avg = data.groupby('hour')['ts'].mean()
    overall_weekly_avg = data.groupby('weekday')['ts'].mean()
    vha_hourly_avg = vha_data.groupby('hour')['ts'].mean()
    community_hourly_avg = community_data.groupby('hour')['ts'].mean()
    vha_weekly_avg = vha_data.groupby('weekday')['ts'].mean()
    community_weekly_avg = community_data.groupby('weekday')['ts'].mean()
    
    # Setting up subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    
    # Plotting average service time by hour of day
    axes[0].plot(overall_hourly_avg.index, overall_hourly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[0].plot(vha_hourly_avg.index, vha_hourly_avg.values, label='VHA', marker='o', linestyle='-', color='blue')
    axes[0].plot(community_hourly_avg.index, community_hourly_avg.values, label='Community Care', marker='o', linestyle='-', color='green')
    axes[0].set_title('Average Service Time by Hour of Day')
    axes[0].set_xlabel('Hour of the Day')
    axes[0].set_ylabel('Average Service Time (ts)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(range(0, 24))  # Set x-axis ticks for every hour of the day
    
    # Plotting average service time by weekday
    axes[1].plot(overall_weekly_avg.index, overall_weekly_avg.values, label='Overall', marker='o', linestyle='-', color='black')
    axes[1].plot(vha_weekly_avg.index, vha_weekly_avg.values, label='VHA', marker='o', linestyle='-', color='blue')
    axes[1].plot(community_weekly_avg.index, community_weekly_avg.values, label='Community Care', marker='o', linestyle='-', color='green')
    axes[1].set_title('Average Service Time by Weekday')
    axes[1].set_xlabel('Day of the Week')
    axes[1].set_ylabel('Average Service Time (ts)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(range(0, 7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])  # Set labels for days of the week
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    data.drop(["hour","weekday"], axis=1, inplace=True)


def standardize_numeric_columns(data, exclude_column='patientsid'):
    """
    Standardizes all numeric columns in a DataFrame except the specified exclude column.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the data to be standardized.
        exclude_column (str, optional): The column name to be excluded from standardization, default is 'patientsid'.
        
    Returns:
        DataFrame: The DataFrame with standardized numeric columns, excluding the specified column.
    """
    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the excluded column name from the list of columns to be standardized
    if exclude_column in numeric_cols:
        numeric_cols.remove(exclude_column)
    
    # Perform standardization only on the selected numeric columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data, scaler

def encode_categorical_columns(data, threshold=10):
    """
    Encodes categorical columns in the DataFrame using One-Hot Encoding if the number of unique categories
    is below a specified threshold, otherwise uses Label Encoding.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
        threshold (int): The threshold to decide between One-Hot Encoding and Label Encoding.
        
    Returns:
        DataFrame: The DataFrame with categorical columns encoded.
    """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Use sparse_output=False for direct dense array output

    for column in data.select_dtypes(include=['object', 'category']).columns:
        unique_values = data[column].nunique()
        if unique_values > threshold:
            # Apply Label Encoding
            data[column] = label_encoder.fit_transform(data[column])
        else:
            # Apply One-Hot Encoding
            # One-hot encode the data
            encoded_data = onehot_encoder.fit_transform(data[[column]])  # Dense array by default due to sparse_output=False
            # Create a DataFrame with encoded data
            encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{i}" for i in range(encoded_data.shape[1])])
            # Concatenate the original DataFrame with the new one
            data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            # Drop the original column as it's now encoded
            data.drop(column, axis=1, inplace=True)

    return data

    
############################################################
#                                                          #
#        ------      Modeling           -------            #
#                                                          #
############################################################ 

    #===============
    #  LSTM Routine
    #===============



############################################################
#                                                          #
#        ------          Evaluation     -------            #
#                                                          #
############################################################ 

