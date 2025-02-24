import pandas as pd

def load_data():
    #loading soil test data
    try:
        df = pd.read_csv('../../datasets/soil_test.csv')
        return df
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'soil_test.csv' is located in the /datasets/ folder.")
        return

def clean_data_remover(data):
    #data_cleaned is for removing the rows which has missing data
    data_cleaned = data.dropna()
    return data_cleaned

def clean_data_filler(data):
    #data_filled is for filling the rows which has missing data
    data_filled = data.fillna(data.mean())
    return data_filled

def clean_data_filter(data,column):
    #Values more than 2 std dev away from the mean is removed from data as an outliner. This process is for a chosen column.
    try:
        mean = data[column].mean()
        print(mean)
        std_dev = data[column].std()
        print(std_dev)
        data_filtered = data[(data[column] >= mean - 3 * std_dev) & (data[column] <= mean + 3 * std_dev)]
        return data_filtered
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

def compute_statistics(data, column):
    #Statistics for a numeric column is computed.
    try:
        min_value = data[column].min()
        max_value = data[column].max()
        mean = data[column].mean()
        median = data[column].median()
        std_dev = data[column].std()
        print(f"Statistics for column '{column}' is below.")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Min: {min_value:.2f}")
        print(f"Max: {max_value:.2f}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

def main():
    data = load_data()
    data = clean_data_filler(data)
    #data = clean_data_remover(data)
    #data = clean_data_filter(data, '..')
    # The Above 2 lines can also be used. For the first line to be used, "data = clean_data_filler(data)" should be removed.
    data = data.round(2)
    print(data)
    compute_statistics(data, 'soil_ph')

if __name__ == '__main__':
    main()
