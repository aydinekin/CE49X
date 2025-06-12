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

## Submission
- **Deadline**: June 13, 2025, 11:59 PM
- Submit your project by pushing your code to your personal GitHub repository
- The repository should include:
  1. Complete code implementation
  2. Documentation
  3. Test files
  4. Sample data and results
- **Important**: After pushing your code to GitHub, send an email to eyuphan.koc@gmail.com with:
  - Your name
  - Your GitHub repository URL

## Grading Rubric

| Category | Excellent (90-100) | Good (80-89) | Fair (70-79) | Needs Improvement (<70) |
|----------|-------------------|--------------|--------------|------------------------|
| Code Quality | Well-structured, documented, efficient | Good structure, adequate documentation | Basic structure, minimal documentation | Poor structure, lacking documentation |
| Functionality | All features implemented, robust | Most features implemented | Basic features implemented | Missing key features |
| Documentation | Comprehensive, clear, professional | Good coverage, clear | Basic coverage | Inadequate coverage |

## Getting Started
1. Clone the starter repository
2. Set up your development environment
3. Review the requirements and documentation
4. Start with basic functionality
5. Incrementally add features
6. Test thoroughly
7. Document as you go

Good luck with your project! Remember to start early and commit your changes regularly. 