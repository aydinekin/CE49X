# Life Cycle Analysis (LCA) Tool

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
