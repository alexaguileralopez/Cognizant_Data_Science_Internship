import pandas as pd
from datetime import datetime

# Load data
def load_csv(path: str):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path (optional): str, relative path of the CSV file

    :return     df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

def convert_to_datetime(df: pd.DataFrame, column: str):
    '''
    This function takes a dataframe and a column and converts a specific 
    column to datetime format.
    
    :param  df: pd.Dataframe, column: string

    :return df: pd.Dataframe
    '''
    
    dummy = df.copy()
    dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
    return dummy

def datetime_hourly(df: pd.DataFrame, column : str):
    '''
    This function restricts the datetime to hours
    
    :param  df: pd.Dataframe, column: string
    :return df: pd.Dataframe
    '''

    dummy = df.copy()
    dummy[column] = dummy[column].dt.floor('H')

    return dummy

def data_merger(sales_df: pd.DataFrame, stock_df: pd.DataFrame, temp_df: pd.DataFrame):
    '''
    This function takes 3 dataframes and merges them aggregating in specific columns to get a 
    single dataframe
    
    :arguments df1,df2,df3: pd.Dataframe
    
    :return merged_df: pd.Dataframe

    '''
    sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
    stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
    temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()

    merged_df = sales_agg.merge(stock_agg, on= ['timestamp', 'product_id'], how= 'right')
    merged_df = merged_df.merge(temp_agg, on= ['timestamp'], how= 'left')
    merged_df['quantity'] = merged_df['quantity'].fillna(0)

    product_categories = sales_df[['product_id', 'category']]
    product_categories = product_categories.drop_duplicates() 

    product_price = sales_df[['product_id', 'unit_price']]
    product_price = product_price.drop_duplicates()

    merged_df = merged_df.merge(product_categories, on= 'product_id', how= 'left')
    merged_df = merged_df.merge(product_price, on= 'product_id', how= 'left')

    return merged_df

def feature_engineering(df: pd.DataFrame):

    '''
    This function converts categorical values into numerical, in order for our ML model to calculate 
    relations with features such as day of the month, week, month, hour. In addition, categories are being 
    transformed into numerical values to get insights into the importance of those when running our model
    
    :arguments  df: pd.Dataframe
    
    :return df: pd.Dataframe
    '''

    df['day_of_month'] = df['timestamp'].dt.day # transforming timestamp into numbers
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df.drop(columns= ['timestamp'], inplace= True) # unnecessary column
    merged_df = pd.get_dummies(df, columns=['category']) # putting numbers into categories
    merged_df.drop(columns= ['product_id'], inplace= True) # unnecessary column

    return merged_df



def load_data() -> pd.DataFrame:

    '''
    This function makes use of all the previous functions and creates aggregate dataframes and merges the data 
    into a single dataframe that will be used for our machine learning model

    :return df:pd.Dataframe
    
    '''

    sales_df = load_csv(path = "TASK 3/Resources/sales.csv")
    stock_df = load_csv(path= "TASK 3/Resources/sensor_stock_levels.csv")
    temp_df = load_csv(path= "TASK 3/Resources/sensor_storage_temperature.csv")

    sales_df = convert_to_datetime(sales_df, 'timestamp')
    stock_df = convert_to_datetime(stock_df, 'timestamp')
    temp_df = convert_to_datetime(temp_df, 'timestamp')

    sales_df = datetime_hourly(sales_df,'timestamp')
    stock_df = datetime_hourly(stock_df, 'timestamp')
    temp_df = datetime_hourly(temp_df, 'timestamp')

    merged_df = data_merger(sales_df, stock_df, temp_df)
    df = feature_engineering(merged_df)

    return df

df = load_data()
print(df.head())
print(df.info())






