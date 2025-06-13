"""
Visualization module for LCA tool.
Handles creation of plots and charts for impact analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import numpy as np

class LCAVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
        self.impact_labels = {
            'carbon_impact': 'Carbon Impact (kg CO2e)',
            'energy_impact': 'Energy Impact (kWh)',
            'water_impact': 'Water Impact (L)',
            'waste_generated_kg': 'Waste Generated (kg)'
        }
    
    def plot_impact_breakdown(self, data: pd.DataFrame, impact_type: str, 
                            group_by: str = 'material_type',
                            title: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing impact breakdown by specified grouping.
        
        Args:
            data: DataFrame with impact data
            impact_type: Type of impact to plot (e.g., 'carbon_impact')
            group_by: Column to group by ('material_type' or 'life_cycle_stage')
            title: Optional title for the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        impact_data = data.groupby(group_by)[impact_type].sum()
        impact_data = impact_data.sort_values(ascending=False)
        ax.pie(impact_data, autopct='%1.1f%%',
               colors=self.colors[:len(impact_data)])
        total = impact_data.sum()
        percentages = (impact_data / total) * 100

        # Create custom legend labels with percentages
        legend_labels = [f"{label}: {pct:.1f}%" for label, pct in zip(impact_data.index, percentages)]
        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.impact_labels[impact_type]} by {group_by.replace("_", " ").title()}')
            
        return fig

    def all_products_comparrison_by_all_stages(self, data: pd.DataFrame, impact: str, unit: str) -> plt.Figure:
        carbon_df = data[['product_id', 'life_cycle_stage', impact]].copy()

        # Pivot to get life cycle stages as separate columns
        pivot_df = carbon_df.pivot(index='product_id', columns='life_cycle_stage', values=impact).fillna(0)

        # Sort columns to ensure consistent stage order
        stage_order = ['manufacturing', 'transportation', 'end-of-life']  # adjust if your stage name differs
        pivot_df = pivot_df[stage_order]

        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot on the created axis
        pivot_df.plot(kind='bar', width=0.75, ax=ax)

        ax.set_title(f"{impact.replace('_', ' ').title()} by Life Cycle Stage for Each Product")
        ax.set_xlabel("Product ID")
        ax.set_ylabel(f"{impact.replace('_', ' ').title()} ({unit})")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Life Cycle Stage")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        return fig
    
    def plot_life_cycle_impacts(self, data: pd.DataFrame, 
                              product_id: str) -> plt.Figure:
        """
        Create a stacked bar chart showing impacts across life cycle stages.
        
        Args:
            data: DataFrame with impact data
            product_id: Product ID to analyze
            
        Returns:
            matplotlib Figure object
        """
        product_data = data[data['product_id'] == product_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        impact_types = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        
        for idx, impact_type in enumerate(impact_types):
            stage_data = product_data.pivot_table(
                index='life_cycle_stage',
                values=impact_type,
                aggfunc='sum'
            )
            
            stage_data.plot(kind='bar', ax=axes[idx], color=self.colors[idx])
            axes[idx].set_title(self.impact_labels[impact_type])
            axes[idx].set_xlabel('Life Cycle Stage')
            axes[idx].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        return fig
    
    def plot_product_comparison(self, data: pd.DataFrame, 
                              product_ids: List[str]) -> plt.Figure:
        """
        Create a radar chart comparing multiple products across impact categories.
        
        Args:
            data: DataFrame with impact data
            product_ids: List of product IDs to compare
            
        Returns:
            matplotlib Figure object
        """
        # Calculate total impacts for each product
        total_impacts = data[data['product_id'].isin(product_ids)].groupby('product_id').agg({
            'carbon_impact': 'sum',
            'energy_impact': 'sum',
            'water_impact': 'sum',
            'waste_generated_kg': 'sum'
        })
        
        # Normalize the data
        normalized = total_impacts.copy()
        for col in total_impacts.columns:
            max_val = total_impacts[col].max()
            if max_val > 0:
                normalized[col] = total_impacts[col] / max_val
        
        # Create radar chart
        categories = list(normalized.columns)
        num_vars = len(categories)
        
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, product_id in enumerate(product_ids):
            values = normalized.loc[product_id].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, linewidth=2, label=product_id)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Product Comparison Across Impact Categories')
        plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
        
        return fig

    def plot_end_of_life_breakdown(self, data: pd.DataFrame, product_id: str) -> plt.Figure:
        product_data = data[data['product_id'] == product_id]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set index to desired x-axis labels
        eol_data = product_data[['life_cycle_stage', 'recycling_rate', 'landfill_rate', 'incineration_rate']]
        eol_data = eol_data.set_index('life_cycle_stage')

        eol_data.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red', 'orange'])

        ax.set_title(f'End-of-Life Management for Product {product_id}')
        ax.set_xlabel('Life Cycle Stage')
        ax.set_ylabel('Rate')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=25)

        return fig

    def plot_impact_correlation(self, data: pd.DataFrame) -> plt.Figure:
        """
        Create a correlation heatmap of different impact categories.
        
        Args:
            data: DataFrame with impact data
            
        Returns:
            matplotlib Figure object
        """
        impact_columns = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        correlation = data[impact_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title('Impact Category Correlations')
        
        return fig 