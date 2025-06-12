"""
Calculations module for LCA tool.
Handles environmental impact calculations and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union
from pathlib import Path

class LCACalculator:
    def __init__(self, impact_factors_path: Union[str, Path] = None):
        """
        Initialize LCA Calculator with impact factors.
        
        Args:
            impact_factors_path: Path to the impact factors JSON file
        """
        self.impact_factors = self._load_impact_factors(impact_factors_path) if impact_factors_path else {}
        
    def _load_impact_factors(self, file_path: Union[str, Path]) -> Dict:
        """Load impact factors from JSON file."""
        from .data_input import DataInput
        data_input = DataInput()
        return data_input.read_impact_factors(file_path)
    
    def calculate_impacts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate environmental impacts for each product and life cycle stage.
        
        Args:
            data: DataFrame containing product data
            
        Returns:
            DataFrame with calculated impacts
        """
        results = []
        
        for _, row in data.iterrows():
            material = row['material_type'].lower()
            stage = row['life_cycle_stage'].lower()
            quantity = row['quantity_kg']
            
            # Get impact factors for the material and stage
            material_factors = self.impact_factors.get(material, {})
            stage_factors = material_factors.get(stage, {})
            
            # Calculate impacts using both direct measurements and impact factors
            impacts = {
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'life_cycle_stage': stage,
                'material_type': material,
                'quantity_kg': quantity,
                
                # Direct measurements from data
                'energy_consumption_kwh': row['energy_consumption_kwh'],
                'transport_distance_km': row['transport_distance_km'],
                'waste_generated_kg': row['waste_generated_kg'],
                
                # Calculated impacts using impact factors
                'carbon_impact': (
                    quantity * stage_factors.get('carbon_impact', 0) +
                    row['carbon_footprint_kg_co2e']
                ),
                'energy_impact': (
                    quantity * stage_factors.get('energy_impact', 0) +
                    row['energy_consumption_kwh']
                ),
                'water_impact': (
                    quantity * stage_factors.get('water_impact', 0) +
                    row['water_usage_liters']
                ),
                
                # End-of-life management
                'recycling_rate': row['recycling_rate'],
                'landfill_rate': row['landfill_rate'],
                'incineration_rate': row['incineration_rate']
            }
            results.append(impacts)
            
        return pd.DataFrame(results)
    
    def calculate_total_impacts(self, impacts: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total impacts across all life cycle stages for each product.
        
        Args:
            impacts: DataFrame with calculated impacts
            
        Returns:
            DataFrame with total impacts per product
        """
        # Group by product and sum impacts
        total_impacts = impacts.groupby(['product_id', 'product_name']).agg({
            'carbon_impact': 'sum',
            'energy_impact': 'sum',
            'water_impact': 'sum',
            'waste_generated_kg': 'sum'
        }).reset_index()
        
        return total_impacts
    
    def normalize_impacts(self, impacts: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize impacts to a common scale (0-1).
        
        Args:
            impacts: DataFrame with calculated impacts
            
        Returns:
            DataFrame with normalized impacts
        """
        normalized = impacts.copy()
        
        impact_columns = ['carbon_impact', 'energy_impact', 'water_impact']
        
        for col in impact_columns:
            max_value = impacts[col].max()
            if max_value > 0:
                normalized[col] = impacts[col] / max_value
                
        return normalized
    
    def compare_alternatives(self, impacts: pd.DataFrame, product_ids: List[str]) -> pd.DataFrame:
        """
        Compare environmental impacts between alternative products.
        
        Args:
            impacts: DataFrame with calculated impacts
            product_ids: List of product IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison = impacts[impacts['product_id'].isin(product_ids)].copy()
        
        # Calculate relative differences
        for impact_type in ['carbon_impact', 'energy_impact', 'water_impact']:
            min_value = comparison[impact_type].min()
            comparison[f'{impact_type}_relative'] = (
                (comparison[impact_type] - min_value) / min_value * 100
            )
            
        return comparison 