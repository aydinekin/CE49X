# Life Cycle Analysis (LCA) Tool

## Project Overview
This LCA tool helps analyze and visualize the environmental impacts of products or processes throughout their entire life cycle. The tool integrates Python programming fundamentals, data science concepts, and environmental science principles to provide comprehensive environmental impact assessment.

## Features

### Data Management
- Support for multiple data formats (CSV, Excel, JSON)
- Comprehensive data validation
- Impact factor database integration
- Life cycle stage tracking

### Impact Analysis
- Carbon footprint calculation
- Energy consumption analysis
- Water usage assessment
- Waste generation tracking
- End-of-life management analysis

### Visualization
- Impact breakdown by material and life cycle stage
- Life cycle impact analysis
- Product comparison using radar charts
- End-of-life management visualization
- Impact category correlation analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lca-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer

# Load data
data_input = DataInput()
product_data = data_input.read_data('data/raw/sample_data.csv')
impact_factors = data_input.read_impact_factors('data/raw/impact_factors.json')

# Calculate impacts
calculator = LCACalculator(impact_factors_path='data/raw/impact_factors.json')
impacts = calculator.calculate_impacts(product_data)

# Visualize results
visualizer = LCAVisualizer()
fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', 'material_type')
```

### Example Notebook
Check out the example notebook in `notebooks/lca_analysis_example.ipynb` for a comprehensive demonstration of the tool's capabilities.

## Data Structure

### Product Data (CSV)
The tool expects product data in CSV format with the following columns:
- `product_id`: Unique identifier for the product
- `product_name`: Name of the product
- `life_cycle_stage`: Stage in the life cycle (Manufacturing, Transportation, End-of-Life)
- `material_type`: Type of material used
- `quantity_kg`: Quantity in kilograms
- `energy_consumption_kwh`: Energy consumption in kilowatt-hours
- `transport_distance_km`: Transportation distance in kilometers
- `transport_mode`: Mode of transportation
- `waste_generated_kg`: Waste generated in kilograms
- `recycling_rate`: Rate of recycling (0-1)
- `landfill_rate`: Rate of landfill disposal (0-1)
- `incineration_rate`: Rate of incineration (0-1)
- `carbon_footprint_kg_co2e`: Carbon footprint in kg CO2e
- `water_usage_liters`: Water usage in liters

### Impact Factors (JSON)
Impact factors are stored in JSON format with the following structure:
```json
{
    "material_name": {
        "life_cycle_stage": {
            "carbon_impact": value,
            "energy_impact": value,
            "water_impact": value
        }
    }
}
```
## Contributions 

### Main Script - final_project.py
final_project.py is a tool that analyzes the environmental impacts with LCA from input to output  in a single script
Code is explained in detail with inline comments.

### Features
- Loads and validates input data
- Performs total and normalized impact calculations
- Computes stage-based environmental impact breakdowns
- Saves all outputs (`.csv` and `.png`) automatically to a `results/` folder
- Produces more than 10 visualizations for comparison and analysis

### Extra Additions 
 - Added new graphs that displays carbon, energy & water impacts that shows and compares all products and all stages in stacked barchart.
The graph is used via visualization.py and its called all_products_comparrison_by_all_stages.
```python
visualizer.all_products_comparrison_by_all_stages(impacts, "carbon_impact", "kg CO2e")
```
 - Changed data validation function where Transportation stage was also accounted in the original but the data is not fitting for that.
```python
  # Validate rates sum to 1
  # Only validate rows where there is waste
  waste_rows = data[data['waste_generated_kg'] > 0]
  rate_columns = ['recycling_rate', 'landfill_rate', 'incineration_rate']
  if not (waste_rows[rate_columns].sum(axis=1) - 1).abs().lt(0.001).all():
      return False
```
 - Made minor changes on existing graph functions, suchs as increasing readability by creating legends and sorting them.

```python
  # Removed labels from on the graph because they were overlapping
  ax.pie(impact_data, autopct='%1.1f%%',
         colors=self.colors[:len(impact_data)])
  
  # Calculate percentages for legends
  total = impact_data.sum()
  percentages = (impact_data / total) * 100
  
  # Create custom legend labels with percentages
  legend_labels = [f"{label}: {pct:.1f}%" for label, pct in zip(impact_data.index, percentages)]
  ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
```
```python
  # Changed x-axis labels
  eol_data = product_data[['life_cycle_stage', 'recycling_rate', 'landfill_rate', 'incineration_rate']]
  eol_data = eol_data.set_index('life_cycle_stage')
  
  eol_data.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red', 'orange'])
  
  ax.set_title(f'End-of-Life Management for Product {product_id}')
  ax.set_xlabel('Life Cycle Stage')
  ax.set_ylabel('Rate')
  ax.set_ylim(0, 1)
  plt.xticks(rotation=25)
```

 - Used plot_impact_breakdown visualization function to plot other type of graphs
```python
  # Plot energy consumption analysis by material type
  fig_energy = visualizer.plot_impact_breakdown(impacts, 'energy_impact', 'material_type')

  # Plot water usage breakdown by material type
  fig_water = visualizer.plot_impact_breakdown(impacts, 'water_impact', 'material_type')

  # Plot waste generation breakdown by material type
  fig_waste = visualizer.plot_impact_breakdown(impacts, 'waste_generated_kg', 'material_type')
```
 - Saved graphs under the file './results/graphs'.
 - Saved impact results under './results' as csv files using utils save_results function.
```python
  save_results(normalized_impacts, './results/normalized_impacts.csv', 'csv')
  save_results(total_impacts, './results/total_impacts.csv', 'csv')
  save_results(impacts_by_stage, './results/impacts_by_stage.csv', 'csv')
```
### Results 
The results of this analysis are saved on the result file path. It consists of the csv files for total_impacts, normalized_impacts and impacts_by_stage. It also consists of graphs in png form for product_comparison, impact_correlation, carbon_impact_breakdown etc. 
Extra features added explained in the above section. 
