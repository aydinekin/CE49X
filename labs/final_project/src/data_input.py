"""
Data input module for LCA tool.
Handles reading and validating input data from various sources.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Union

class DataInput:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.required_columns = [
            'product_id', 'product_name', 'life_cycle_stage', 'material_type',
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'transport_mode', 'waste_generated_kg', 'recycling_rate',
            'landfill_rate', 'incineration_rate', 'carbon_footprint_kg_co2e',
            'water_usage_liters'
        ]
    
    def read_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read data from various file formats.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            DataFrame containing the input data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            return pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
            
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data structure and content.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check required columns
        if not all(col in data.columns for col in self.required_columns):
            return False
            
        # Validate numeric columns
        numeric_columns = [
            'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
            'waste_generated_kg', 'recycling_rate', 'landfill_rate',
            'incineration_rate', 'carbon_footprint_kg_co2e', 'water_usage_liters'
        ]
        
        for col in numeric_columns:
            if not pd.to_numeric(data[col], errors='coerce').notnull().all():
                return False
                
        # Validate rates sum to 1
        rate_columns = ['recycling_rate', 'landfill_rate', 'incineration_rate']
        if not (data[rate_columns].sum(axis=1) - 1).abs().lt(0.001).all():
            return False
            
        return True
        
    def read_impact_factors(self, file_path: Union[str, Path]) -> Dict:
        """
        Read impact factors from JSON file.
        
        Args:
            file_path: Path to the impact factors JSON file
            
        Returns:
            Dictionary containing impact factors
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Impact factors file not found: {file_path}")
            
        if file_path.suffix != '.json':
            raise ValueError("Impact factors must be provided in JSON format")
            
        with open(file_path, 'r') as f:
            impact_factors = json.load(f)
            
        return impact_factors 