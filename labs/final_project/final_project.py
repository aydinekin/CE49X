
import pandas as pd
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer
from src.utils import save_results

def main():
    # Initialize data input handler
    data_input = DataInput()

    # Load product data
    product_data = data_input.read_data('data/raw/sample_data.csv')
    print("Product Data Shape:", product_data.shape)
    print("Product Data Head: ",product_data.head())

    # Check for validity of the data
    data_valid = data_input.validate_data(product_data)
    if(data_valid):
        print("Data is valid. It contains all of the required fields.")
    else:
        print("Data is not valid")
        return

    # Load impact factors
    impact_factors = data_input.read_impact_factors('data/raw/impact_factors.json')
    print("Available Materials:", list(impact_factors.keys()))
    print("\nImpact Factors for Steel:", pd.DataFrame(impact_factors['steel']))

    # Initialize calculator
    calculator = LCACalculator(impact_factors_path='data/raw/impact_factors.json')

    # Calculate impacts
    impacts = calculator.calculate_impacts(product_data)
    print("Calculated Impacts Shape:", impacts.shape)
    print("Calculated Impacts Shape: ", impacts.head())

    # Impacts by stage
    impacts_by_stage = impacts.groupby(['product_id', 'life_cycle_stage']).sum().reset_index()

    # Calculate total impacts
    total_impacts = calculator.calculate_total_impacts(impacts)
    print("Total Impacts by Product: ", total_impacts)

    # Initialize LCAVisualizer
    visualizer = LCAVisualizer()

    # Plot carbon impact breakdown by material type
    fig_carbon = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', 'material_type')

    # Plot energy consumption analysis by material type
    fig_energy = visualizer.plot_impact_breakdown(impacts, 'energy_impact', 'material_type')

    # Plot water usage breakdown by material type
    fig_water = visualizer.plot_impact_breakdown(impacts, 'water_impact', 'material_type')

    # Plot waste generation breakdown by material type
    fig_waste = visualizer.plot_impact_breakdown(impacts, 'waste_generated_kg', 'material_type')

    # Plot life cycle impacts for Product1
    fig_life_cycle = visualizer.plot_life_cycle_impacts(impacts, 'P001')

    # Compare two products
    fig_comparrison = visualizer.plot_product_comparison(impacts, ['P001', 'P002'])

    # Plot end-of-life breakdown for Product1
    fig_end_of_life = visualizer.plot_end_of_life_breakdown(impacts, 'P007')

    # Plot impact correlations
    fig_impact_correlation = visualizer.plot_impact_correlation(impacts)

    # PLot carbon impact for all products on all stages
    fig_carbon_all = visualizer.all_products_comparrison_by_all_stages(impacts, "carbon_impact", "kg CO2e")

    # PLot energy impact for all products on all stages
    fig_energy_all = visualizer.all_products_comparrison_by_all_stages(impacts, "energy_impact", "MJ")

    # PLot water impact for all products on all stages
    fig_water_all = visualizer.all_products_comparrison_by_all_stages(impacts, "water_impact", "L")

    # Save the generated graphs
    fig_carbon.savefig("./results/graphs/carbon_impact_breakdown.png")
    fig_energy.savefig("./results/graphs/energy_analysis.png")
    fig_life_cycle.savefig("./results/graphs/life_cycle_impacts.png")
    fig_water.savefig("./results/graphs/water_impacts.png")
    fig_waste.savefig("./results/graphs/waste_generated.png")
    fig_comparrison.savefig("./results/graphs/product_comparison.png")
    fig_end_of_life.savefig("./results/graphs/end_of_life_breakdown.png")
    fig_impact_correlation.savefig("./results/graphs/impact_correlation.png")
    fig_carbon_all.savefig("./results/graphs/all_products_carbon_impact")
    fig_energy_all.savefig("./results/graphs/all_products_energy_impact")
    fig_water_all.savefig("./results/graphs/all_products_water_impact")

    # Display the generated graphs
    #plt.show()

    # Normalize impacts for comparison
    normalized_impacts = calculator.normalize_impacts(impacts)
    print("Normalized Impacts: ", normalized_impacts.head())

    # Save the impact results as csv
    save_results(normalized_impacts, './results/normalized_impacts.csv', 'csv')
    save_results(total_impacts, './results/total_impacts.csv', 'csv')
    save_results(impacts_by_stage, './results/impacts_by_stage.csv', 'csv')

    # Compare alternative products
    comparison = calculator.compare_alternatives(normalized_impacts, ['P001', 'P002'])
    print("Product Comparison: ", comparison)

if __name__ == '__main__':
    main()

