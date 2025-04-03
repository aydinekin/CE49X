<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import seaborn as sns

def load_data(file_path):
    # Loading data from the specified file path
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The dataset file '{file_path}' was not found. Please ensure it is located in the correct folder.")
        return

def data_cleaner(data):
    #data_cleaned is for removing the rows which has missing data
    data_cleaned = data.dropna()
    return data_cleaned

def display_dataset_info(df, city_name):
    # Display basic information about the dataset
    print(f"Dataset information for {city_name}:")
    print(df.info())
    print("\n")

def display_summary_statistics(df, city_name):
    # Display summary statistics of the dataset
    print(f"Summary statistics for {city_name}:")
    print(df.describe())
    print("\n")


def calculate_wind_speed(df):
    # Calculate wind speed from u10m and v10m components using Pythagorean theorem
    if 'u10m' not in df.columns or 'v10m' not in df.columns:
        raise ValueError("DataFrame must contain 'u10m' and 'v10m' columns")

    wind_speed = np.sqrt(df['u10m'] ** 2 + df['v10m'] ** 2)
    print("\n")
    return wind_speed

def calculate_monthly_averages(df):
    # Calculate monthly averages of wind speed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_speed'] = calculate_wind_speed(df)
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_averages = df.groupby('month').agg({'wind_speed': 'mean'}).reset_index()
    print(monthly_averages)
    print("\n")
    return monthly_averages

def assign_season(date):
    # Assign season based on the date
    if pd.Timestamp(year=date.year, month=3, day=21) <= date < pd.Timestamp(year=date.year, month=6, day=21):
        return 'Spring'
    elif pd.Timestamp(year=date.year, month=6, day=21) <= date < pd.Timestamp(year=date.year, month=9, day=23):
        return 'Summer'
    elif pd.Timestamp(year=date.year, month=9, day=23) <= date < pd.Timestamp(year=date.year, month=12, day=21):
        return 'Fall'
    else:
        return 'Winter'

def calculate_seasonal_averages(df):
    # Calculate seasonal averages of wind speed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_speed'] = calculate_wind_speed(df)
    df['season'] = df['timestamp'].apply(assign_season)
    seasonal_averages = df.groupby('season').agg({'wind_speed': 'mean'}).reset_index()
    print(seasonal_averages)
    print("\n")
    return seasonal_averages

def compare_seasonal_patterns(berlin_seasonal_averages, munich_seasonal_averages):
    # Compare seasonal wind speed patterns between Berlin and Munich
    comparison_df = pd.merge(berlin_seasonal_averages, munich_seasonal_averages, on='season', suffixes=('_Berlin', '_Munich'))
    print(comparison_df)
    print("\n")
    return comparison_df

def identify_extreme_weather(df, wind_speed):
    # Identify days with extreme weather based on wind speed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_speed'] = wind_speed
    df['date'] = df['timestamp'].dt.date
    daily_max_wind_speed = df.groupby('date')['wind_speed'].max().reset_index()
    extreme_days = daily_max_wind_speed[daily_max_wind_speed['wind_speed'] == daily_max_wind_speed['wind_speed'].max()]
    print(extreme_days)
    print("\n")
    return extreme_days

def calculate_diurnal_patterns(df):
    # Calculate diurnal (daily) patterns in wind speed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['wind_speed'] = calculate_wind_speed(df)
    df['hour'] = df['timestamp'].dt.hour
    diurnal_patterns = df.groupby('hour').agg({'wind_speed': 'mean'}).reset_index()
    print(diurnal_patterns)
    print("\n")
    return diurnal_patterns

def plot_monthly_averages(berlin_monthly_averages, munich_monthly_averages):
    # Plot monthly average wind speeds for Berlin and Munich
    plt.figure(figsize=(12, 6))
    plt.plot(berlin_monthly_averages['month'].astype(str), berlin_monthly_averages['wind_speed'], label='Berlin', marker='o')
    plt.plot(munich_monthly_averages['month'].astype(str), munich_monthly_averages['wind_speed'], label='Munich', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.title('Monthly Average Wind Speeds for Berlin and Munich')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_seasonal_comparison(berlin_seasonal_averages, munich_seasonal_averages):
    # Plot seasonal average wind speeds for Berlin and Munich
    comparison_df = pd.merge(berlin_seasonal_averages, munich_seasonal_averages, on='season', suffixes=('_Berlin', '_Munich'))
    comparison_df.set_index('season', inplace=True)
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Season')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.title('Seasonal Average Wind Speeds for Berlin and Munich')
    plt.legend(['Berlin', 'Munich'])
    plt.grid(True)
    plt.show()

def calculate_wind_direction(u, v):
    # Calculate wind direction from u10m and v10m components using arctangent function
    wind_direction = (np.arctan2(v, u) * 180 / np.pi + 360) % 360
    return wind_direction

def plot_wind_rose(df, city_name):
    # Plot wind rose diagram for the given city
    df['wind_direction'] = calculate_wind_direction(df['u10m'], df['v10m'])
    df['wind_speed'] = calculate_wind_speed(df)
    ax = WindroseAxes.from_ax()
    ax.bar(df['wind_direction'], df['wind_speed'], normed=True, opening=0.8, edgecolor='white')
    ax.set_title(f'Wind Rose Diagram for {city_name}')
    ax.set_legend()
    plt.show()

def main():
    # Define file paths for Berlin and Munich datasets
    berlin_era5_file_path = '../../datasets/berlin_era5_wind_20241231_20241231.csv'
    munich_era5_file_path = '../../datasets/munich_era5_wind_20241231_20241231.csv'

    # Load Berlin dataset
    berlin_data = load_data(berlin_era5_file_path)
    # Display basic information about Berlin dataset
    display_dataset_info(berlin_data, "Berlin")
    # Clean Berlin dataset by removing rows with missing data
    berlin_data = data_cleaner(berlin_data)
    # Display summary statistics for Berlin dataset
    display_summary_statistics(berlin_data, "Berlin")
    # Calculate wind speed for Berlin dataset
    print(f"Wind Speed for Berlin")
    berlin_wind_speed = calculate_wind_speed(berlin_data)
    print(berlin_wind_speed)
    # Calculate monthly averages for Berlin dataset
    print(f"Monthly Averages of Wind Speed for Berlin")
    berlin_monthly_averages = calculate_monthly_averages(berlin_data)
    # Calculate seasonal averages for Berlin dataset
    print(f"Seasonal Averages of Wind Speed for Berlin")
    berlin_seasonal_averages = calculate_seasonal_averages(berlin_data)
    # Identify extreme weather days for Berlin
    print(f"Extreme Days for Berlin")
    berlin_extreme_days = identify_extreme_weather(berlin_data, berlin_wind_speed)
    # Day with the highest wind speed is 16.12.2024 in Berlin and it is roughly 8.5 m/s. When I check online resources,
    # I see that the maximum wind speed in that day is around 7.5 m/s which equates to roughly 16.8mph.
    # Calculate diurnal patterns for Berlin dataset
    print(f"Diuarnal Patterns in Wind Speed for Berlin")
    berlin_diurnal_patterns = calculate_diurnal_patterns(berlin_data)


    # Load Munich dataset
    munich_data = load_data(munich_era5_file_path)
    # Display basic information about Munich dataset
    display_dataset_info(munich_data, "Munich")
    # Clean Munich dataset by removing rows with missing data
    munich_data = data_cleaner(munich_data)
    # Display summary statistics for Munich dataset
    display_summary_statistics(munich_data, "Munich")
    # Calculate wind speed for Munich dataset
    print(f"Wind Speed for Munich")
    munich_wind_speed = calculate_wind_speed(munich_data)
    print(munich_wind_speed)
    # Calculate monthly averages for Munich dataset
    print(f"Monthly Averages of Wind Speed for Munich")
    munich_monthly_averages = calculate_monthly_averages(munich_data)
    # Calculate seasonal averages for Munich dataset
    print(f"Seasonal Averages of Wind Speed for Munich")
    munich_seasonal_averages = calculate_seasonal_averages(munich_data)
    # Identify extreme weather days for Munich
    print(f"Extreme Days for Munich")
    munich_extreme_days = identify_extreme_weather(munich_data, munich_wind_speed)
    # Day with the highest wind speed is 06.12.2024 in Munich and it is roughly 9 m/s. When I check online resources,
    # I see that the maximum wind speed in that day is around 6.7 m/s which equates to roughly 15mph.
    # Calculate diurnal patterns for Munich dataset
    print(f"Diurnal Patterns in Wind Speed for Munich")
    munich_diurnal_patterns = calculate_diurnal_patterns(munich_data)

    # Compare seasonal patterns between Berlin and Munich
    print(f"Seasonal Pattern Comparison of Berlin and Munich")
    seasonal_comparrison = compare_seasonal_patterns(berlin_seasonal_averages, munich_seasonal_averages)
    # Time series plot of monthly average wind speeds for both cities. Call the function with the calculated monthly averages
    plot_monthly_averages(berlin_monthly_averages, munich_monthly_averages)
    # Seasonal comparison bar charts. Call the function with the calculated seasonal averages
    plot_seasonal_comparison(berlin_seasonal_averages, munich_seasonal_averages)
    # Wind rose diagrams or directional analysis. Call the function for both Berlin and Munich datasets
    plot_wind_rose(berlin_data, 'Berlin')
    plot_wind_rose(munich_data, 'Munich')

if __name__ == '__main__':
    main()


