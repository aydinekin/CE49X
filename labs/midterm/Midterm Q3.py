# Test data for Question 3
lengths = [3.5, 6.2, 4.8, 7.1]


# Your solution here
def process_beam_lengths(lengths):
    # Convert each length to feet
    lengths_in_feet = [length * 3.28084 for length in lengths] #All the lengths in length is multiplied to turn the feet by list comprehension.
    # Filter lengths greater than 5 meters
    greater_than_5m = [length for length in lengths if length > 5] #same with condition
    return (lengths_in_feet, greater_than_5m)


# Test your solution
result = process_beam_lengths(lengths)
print(f"Lengths in feet: {result[0]}")
print(f"Lengths > 5m: {result[1]}")