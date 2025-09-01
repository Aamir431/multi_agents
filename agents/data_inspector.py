# agents/data_inspector.py
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from base_agent import BaseAgent, AgentConfig, PipelineState

class ColumnTypeDetector(BaseAgent):
    """
    A class to automatically detect column types (numerical, categorical, datetime) in a pandas DataFrame.
    """

    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)
        # Config se categorical_threshold lelo (default 0.05)
        self.categorical_threshold = getattr(self.config, 'categorical_threshold', 0.05)

    def run(self) -> str:
        """
        Perform data inspection and column type detection.
        """
        self.logger.info("Starting data inspection...")
        
        # Config se parameters lo
        sample_size = self.config.sample_size
        detect_column_types = self.config.detect_column_types
        generate_summary = self.config.generate_summary
        
        data = self.state.raw_df
        
        if data is None:
            raise ValueError("No data available for inspection. Run DataLoadingAgent first.")
        
        # Sample data if configured
        if sample_size > 0 and len(data) > sample_size:
            data_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
        else:
            data_sample = data
        
        message_parts = []
        
        # Perform column type detection if configured
        if detect_column_types:
            schema = self.detect_column_types(data_sample)
            self.state.schema = schema  # Store in shared state
            message_parts.append(f"Column types detected: {len(schema)} columns")
        
        # Generate summary if configured
        if generate_summary:
            summary = self._generate_summary(data_sample)
            message_parts.append("Data summary generated")
        
        message = f"Inspected data with shape: {data.shape}"
        if message_parts:
            message += " | " + " | ".join(message_parts)
        
        self.logger.info(message)
        return message

    def detect_column_types(self, df: pd.DataFrame) -> dict:
        """
        Detects the type of each column in a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            dict: A dictionary mapping each column name to its detected type
                  ('numerical', 'categorical', 'datetime').
        """
        schema = {}
        total_rows = len(df)

        for column in df.columns:
            col_data = df[column].dropna()  # Drop NAs for better inference
            unique_count = col_data.nunique()
            unique_ratio = unique_count / total_rows if total_rows > 0 else 0

            # 1. Check for Datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                schema[column] = 'datetime'
                continue

            # 2. Try parsing object-type columns as datetime
            is_date_column = False
            if col_data.dtype == 'object':
                try:
                    parsed_dates = pd.to_datetime(col_data, errors='raise', utc=True)
                    if parsed_dates.min().year < 1900:
                        raise ValueError("Dates are too old, likely parsed from integers.")
                    is_date_column = True
                except (TypeError, ValueError, pd.errors.ParserError):
                    is_date_column = False

            if is_date_column:
                schema[column] = 'datetime'
                continue

            # 3. Check for Categorical
            if unique_ratio <= self.categorical_threshold:
                schema[column] = 'categorical'
            elif col_data.dtype == 'object':
                schema[column] = 'categorical'  # Text data
            # 4. Check for Numerical
            elif is_numeric_dtype(col_data):
                schema[column] = 'numerical'
            else:
                schema[column] = 'unknown'

        return schema

    def _generate_summary(self, data):
        """Generate data summary"""
        return data.describe()

# Optional: Agar tumhe alag se detect_column_types call karna hai
# Add this at the end of data_inspector.py
def inspect_data(df: pd.DataFrame, categorical_threshold: float = 0.05) -> dict:
    """
    Standalone function to detect column types without requiring AgentConfig and PipelineState.
    """
    schema = {}
    total_rows = len(df)

    for column in df.columns:
        col_data = df[column].dropna()
        unique_count = col_data.nunique()
        unique_ratio = unique_count / total_rows if total_rows > 0 else 0

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            schema[column] = 'datetime'
            continue

        # Check for categorical
        if unique_ratio <= categorical_threshold:
            schema[column] = 'categorical'
        elif col_data.dtype == 'object':
            schema[column] = 'categorical'
        # Check for numerical
        elif pd.api.types.is_numeric_dtype(col_data):
            schema[column] = 'numerical'
        else:
            schema[column] = 'unknown'

    return schema