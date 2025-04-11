# Data for Question 4
measurements = [
    {'site': 'A', 'depth': 2.5, 'soil_type': 'clay'},
    {'site': 'B', 'depth': 3.8, 'soil_type': 'sand'},
    {'site': 'C', 'depth': 1.9, 'soil_type': 'clay'},
    {'site': 'D', 'depth': 4.2, 'soil_type': 'gravel'}
]

# Your solution here
# 1. Find the average depth for clay soil sites
# We create a list of depths where the soil type is 'clay'
clay_depths = [m['depth'] for m in measurements if m['soil_type'] == 'clay']
# Calculate the average of those depths (if any exist, to avoid division by zero)
average_clay_depth = sum(clay_depths) / len(clay_depths) if clay_depths else 0

# 2. Create a list of site names where depth is greater than 3 meters
# We check each measurement and include the site name if depth > 3
deep_sites = [m['site'] for m in measurements if m['depth'] > 3]

# 3. Count how many different soil types are present
# Use a set to automatically remove duplicates, then count the unique types
soil_types = {m['soil_type'] for m in measurements}  # set ensures each soil type appears only once
soil_type_count = len(soil_types)

print(f"Average depth for clay soil sites: {average_clay_depth:.2f} meters")
print(f"Sites with depth > 3m: {deep_sites}")
print(f"Number of different soil types: {soil_type_count}")