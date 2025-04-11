# Test data for Question 1
test_measurements = [5.2, 12.8, 3.1, 15.6, 8.9, 0.0, 4.5]

# Your solution here
def analyze_rainfall(measurements):
    avg = sum(measurements) / len(measurements) #average is taken
    maximum = max(measurements)
    count = 0
    for i in measurements:    #days above 10mm, is taken with a code of block with for loop
        if i > 10:
            count += 1
    return round(avg, 2), maximum, count
    pass

# Test your solution
result = analyze_rainfall(test_measurements)
print(f"Average rainfall: {result[0]:.2f} mm")
print(f"Maximum rainfall: {result[1]} mm")
print(f"Days above 10mm: {result[2]}")