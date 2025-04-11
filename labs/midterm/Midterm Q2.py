# Data for Question 2
materials = {
    'steel': {'density': 7850, 'elastic_modulus': 200e9, 'yield_strength': 250e6},
    'concrete': {'density': 2400, 'elastic_modulus': 30e9, 'yield_strength': 30e6},
    'wood': {'density': 500, 'elastic_modulus': 12e9, 'yield_strength': 40e6}
}

# 1. Print the material with the highest density
# Using max() with a lambda to find the material with the highest 'density' value
highest_density_mat = max(materials, key=lambda m: materials[m]['density'])  # lambda is a function that we don't reuse.
print(highest_density_mat)

# 2. Calculate the average elastic modulus of all materials
# Summing all 'elastic_modulus' values and dividing by the number of materials
total = sum(materials[m]['elastic_modulus'] for m in materials)
avg_elastic_modulus = total / len(materials)
print(f"Average elastic modulus: {avg_elastic_modulus:.2f}")

# 3. Create a new dictionary containing only materials with yield strength greater than 35e6
# materials filetered based on yield strength and created a new dictionary.
strong_materials = {
    name: props
    for name, props in materials.items()
    if props['yield_strength'] > 35e6
}

print("Materials with yield strength > 35e6:", strong_materials)