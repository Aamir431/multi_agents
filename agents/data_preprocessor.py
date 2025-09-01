import sys
sys.path.append('/Users/apple/Desktop/multi_agents/agents')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from base_agent import BaseAgent, AgentConfig, PipelineState

class DataPreprocessor(BaseAgent):
    """
    A class to preprocess a pandas DataFrame.
    Handles missing values, outliers, scaling, and categorical encoding.
    Preserves the target variable from transformations.
    """

    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)
        # Config se parameters lo
        self.handle_missing = self.config.handle_missing
        self.scale_features = self.config.scale_features
        self.encode_categorical = self.config.encode_categorical
        self.remove_outliers = self.config.remove_outliers
        
        # Get target column from config - ADDED
        self.target_column = getattr(self.config, 'target_column', None)
        if not self.target_column:
            # Fallback to old config name for backward compatibility
            self.target_column = getattr(self.config, 'target_col', None)
        
        # Initialize other attributes
        self.imputation_values = {}
        self.known_categories = {}
        self.outlier_boundaries = {}
        self.scaler = None
        self.encoders = {}
        
        # Initialize numerical_cols and categorical_cols as empty lists
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_columns = []  # ADDED: Track feature columns separately from target
        
    def run(self) -> str:
        """
        Main method to run data preprocessing.
        """
        self.logger.info("Starting data preprocessing...")
        
        if self.state.raw_df is None:
            raise ValueError("No data available for preprocessing. Run DataLoadingAgent first.")
        
        if self.state.schema is None:
            raise ValueError("No schema available. Run ColumnTypeDetector first.")
        
        # Get schema from shared state
        schema = self.state.schema
        
        # ADDED: Identify and validate target column
        if self.target_column:
            if self.target_column not in self.state.raw_df.columns:
                available_cols = self.state.raw_df.columns.tolist()
                raise ValueError(
                    f"Target column '{self.target_column}' not found in data. "
                    f"Available columns: {available_cols}"
                )
            self.logger.info(f"Target column identified: {self.target_column}")
        
        # Separate target column from features - ADDED
        all_columns = list(schema.keys())
        if self.target_column and self.target_column in all_columns:
            # Remove target column from feature lists
            all_columns.remove(self.target_column)
        
        self.numerical_cols = [col for col in all_columns if schema.get(col) == 'numerical']
        self.categorical_cols = [col for col in all_columns if schema.get(col) == 'categorical']
        self.feature_columns = self.numerical_cols + self.categorical_cols  # ADDED
        
        # Fit and transform the data
        self.fit(self.state.raw_df)
        
        #  Process the ENTIRE dataframe, not just features
        processed_df = self.transform(self.state.raw_df)
        
        # Store processed data in shared state
        self.state.processed_df = processed_df
        self.state.target_column = self.target_column  # ADDED: Store target column in state
        
        message = f"Preprocessing completed. Original shape: {self.state.raw_df.shape}, Processed shape: {processed_df.shape}"
        self.logger.info(message)
        return message

    def _clean_categorical_features(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        """Helper function to clean categorical columns."""
        if not self.encode_categorical:
            return df
            
        df_copy = df.copy()
        for col in self.categorical_cols:
            if col in df_copy.columns:
                # Convert to string, lowercase, and strip whitespace
                df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
                # Standardize 'nan' strings
                df_copy[col] = df_copy[col].replace('nan', np.nan)
                # Impute missing values if enabled
                if self.handle_missing:
                    df_copy[col] = df_copy[col].fillna(self.imputation_values.get(col))
                if not is_fit:  # During transform, handle unseen categories
                    known_cats = self.known_categories.get(col, set())
                    df_copy[col] = df_copy[col].apply(lambda x: x if x in known_cats else 'unknown')
        return df_copy
    
    def fit(self, df: pd.DataFrame):
        """Learns all preprocessing parameters from the training data."""
        df_copy = df.copy()

        # WORK ONLY ON FEATURE COLUMNS (exclude target) - ADDED
        working_df = df_copy[self.feature_columns].copy() if self.feature_columns else df_copy.copy()

        # 1. Learn imputation values if enabled
        if self.handle_missing:
            for col in self.numerical_cols:
                if col in working_df.columns:
                    self.imputation_values[col] = working_df[col].mean()
            for col in self.categorical_cols:
                if col in working_df.columns:
                    normalized_series = working_df[col].dropna().astype(str).str.lower().str.strip()
                    if not normalized_series.empty:
                        self.imputation_values[col] = normalized_series.mode()[0]
                        self.known_categories[col] = set(normalized_series.unique())
                        self.known_categories[col].add('unknown')
                    else:
                        self.imputation_values[col] = 'missing'
                        self.known_categories[col] = {'unknown'}
        
        # 2. Learn outlier boundaries and scaler if enabled
        if self.remove_outliers and self.numerical_cols:
            numerical_df = working_df[self.numerical_cols].copy()
            for col in self.numerical_cols:
                if self.handle_missing:
                    numerical_df[col] = numerical_df[col].fillna(self.imputation_values[col])
            for col in self.numerical_cols:
                Q1, Q3 = numerical_df[col].quantile(0.25), numerical_df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_boundaries[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        # 3. Learn scaler if enabled
        if self.scale_features and self.numerical_cols:
            numerical_df = working_df[self.numerical_cols].copy()
            for col in self.numerical_cols:
                if self.handle_missing:
                    numerical_df[col] = numerical_df[col].fillna(self.imputation_values[col])
                if self.remove_outliers:
                    lower, upper = self.outlier_boundaries.get(col, (None, None))
                    if lower is not None and upper is not None:
                        numerical_df[col] = numerical_df[col].clip(lower=lower, upper=upper)
            self.scaler = StandardScaler()
            self.scaler.fit(numerical_df)
            
        # 4. Learn encoders if enabled - CRITICAL FIX: Ensure ALL categorical columns get encoders
        if self.encode_categorical:
            cleaned_categorical_df = self._clean_categorical_features(working_df, is_fit=True)
            for col in self.categorical_cols:
                if col in cleaned_categorical_df.columns:
                    # Check if column still exists (might have been dropped during cleaning)
                    if col not in cleaned_categorical_df.columns:
                        continue
                        
                    unique_vals = cleaned_categorical_df[col].nunique()
                    if unique_vals == 2:
                        encoder = LabelEncoder()
                        # Handle case where there might be NaN values during fitting
                        non_null_data = cleaned_categorical_df[col].dropna()
                        if len(non_null_data) > 0:
                            encoder.fit(non_null_data)
                            self.encoders[col] = encoder
                    elif unique_vals > 2:
                        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        # Handle potential NaN values
                        non_null_data = cleaned_categorical_df[[col]].dropna()
                        if len(non_null_data) > 0:
                            encoder.fit(non_null_data)
                            self.encoders[col] = encoder
                    else:  # Only 1 unique value or all NaN
                        # Create a dummy encoder that will return constant values
                        encoder = LabelEncoder()
                        # Fit with a dummy value to avoid errors
                        encoder.fit(['dummy_value'])
                        self.encoders[col] = encoder
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all learned transformations to a DataFrame."""
        df_copy = df.copy()

        #  Process ALL columns initially
        result_df = df_copy.copy()

        # --- Process Feature Columns ---
        if self.feature_columns:
            # WORK ONLY ON FEATURE COLUMNS (exclude target)
            working_df = df_copy[self.feature_columns].copy()
            processed_features = working_df.copy()

            # --- Clean and Scale Numerical Features ---
            if self.numerical_cols:
                numerical_df = working_df[self.numerical_cols].copy()
                
                # 1. Impute if enabled
                if self.handle_missing:
                    for col in self.numerical_cols:
                        numerical_df[col] = numerical_df[col].fillna(self.imputation_values.get(col))
                
                # 2. Cap outliers if enabled
                if self.remove_outliers:
                    for col in self.numerical_cols:
                        lower, upper = self.outlier_boundaries.get(col, (None, None))
                        if lower is not None and upper is not None:
                            numerical_df[col] = numerical_df[col].clip(lower=lower, upper=upper)
                
                # 3. Scale if enabled
                if self.scale_features and self.scaler:
                    scaled_values = self.scaler.transform(numerical_df)
                    processed_features[self.numerical_cols] = scaled_values
                else:
                    processed_features[self.numerical_cols] = numerical_df
            
            # --- Clean and Encode Categorical Features ---
            if self.encode_categorical:
                cleaned_categorical_df = self._clean_categorical_features(working_df, is_fit=False)
                
                # First handle all categorical encoding
                columns_to_drop = []
                for col in self.categorical_cols:
                    if col in cleaned_categorical_df.columns and col in self.encoders:
                        encoder = self.encoders[col]
                        
                        # Handle missing values before encoding
                        col_data = cleaned_categorical_df[col].copy()
                        if self.handle_missing and col_data.isna().any():
                            col_data = col_data.fillna(self.imputation_values.get(col, 'missing'))
                        
                        if isinstance(encoder, LabelEncoder):
                            # For LabelEncoder, handle unseen labels gracefully
                            try:
                                encoded_values = encoder.transform(col_data)
                                processed_features[col] = encoded_values
                            except ValueError:
                                # Handle unseen labels by mapping them to a default value
                                unique_labels = set(encoder.classes_)
                                processed_features[col] = col_data.apply(lambda x: encoder.transform([x])[0] if x in unique_labels else 0)
                        
                        elif isinstance(encoder, OneHotEncoder):
                            # For OneHotEncoder, handle transformation
                            try:
                                encoded_data = encoder.transform(cleaned_categorical_df[[col]])
                                encoded_df = pd.DataFrame(encoded_data, 
                                                        columns=encoder.get_feature_names_out([col]), 
                                                        index=processed_features.index)
                                processed_features = pd.concat([processed_features.drop(col, axis=1), encoded_df], axis=1)
                            except Exception as e:
                                self.logger.warning(f"Error encoding column {col}: {e}")
                                # If encoding fails, drop the column to avoid string values
                                columns_to_drop.append(col)
                
                # Drop any categorical columns that couldn't be encoded properly
                processed_features = processed_features.drop(columns=[col for col in columns_to_drop if col in processed_features.columns])
            
            # FINAL SANITY CHECK: Ensure all feature columns are numeric
            for col in processed_features.columns:
                if processed_features[col].dtype == 'object':
                    self.logger.warning(f"Column {col} is still categorical after encoding. Attempting to convert to numeric.")
                    try:
                        processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')
                        # Fill any remaining NaN values with 0
                        processed_features[col] = processed_features[col].fillna(0)
                    except:
                        self.logger.error(f"Could not convert column {col} to numeric. Dropping column.")
                        processed_features = processed_features.drop(columns=[col])
            
            # Update result_df with processed features
            result_df = processed_features
            
            # Add target column back to processed data
            if self.target_column and self.target_column in df_copy.columns:
                target_data = df_copy[self.target_column].copy()
                
                # If target is categorical, encode it
                schema = self.state.schema
                if schema and schema.get(self.target_column) == 'categorical':
                    unique_targets = target_data.unique()
                    if len(unique_targets) <= 2:  # Binary classification
                        le = LabelEncoder()
                        result_df[self.target_column] = le.fit_transform(target_data)
                        # Store the mapping for reference
                        self.state.target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    else:
                        # For multi-class, use simple integer encoding
                        result_df[self.target_column] = pd.factorize(target_data)[0]
                else:
                    # For numerical target, just copy it
                    result_df[self.target_column] = target_data
        
        return result_df