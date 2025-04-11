# Data for Question 6
import math

loads = [25.5, 30.2, 18.7, 42.1, 28.9, 35.6]

# Your solution here
# 1. Calculate the standard deviation of the loads
mean_load = sum(loads) / len(loads)
std_dev = math.sqrt(sum((x - mean_load) ** 2 for x in loads) / len(loads))
print(std_dev)
# 2. Find the load value closest to the mean
closest_to_mean = min(loads, key = lambda x: abs(x - mean_load))
print(closest_to_mean)
# 3. Create a new list containing only loads that are within ±10% of the mean
tolerance = 0.10 * mean_load
within_10_percent = [x for x in loads if abs(x - mean_load) <= tolerance]
print(within_10_percent)
# Note: You can use the following formula for standard deviation:
# std_dev = math.sqrt(sum((x - mean)**2 for x in data) / len(data))