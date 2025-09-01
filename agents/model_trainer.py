import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

from data_inspector import inspect_data
from data_preprocessor import DataPreprocessor
from base_agent import BaseAgent, AgentConfig, PipelineState

class ModelTrainer(BaseAgent):
    """
    A class to handle the end-to-end process of training and evaluating a machine learning model.
    """
    
    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)
        
        #  SAFE: Use getattr() with default values for all parameters
        # ADDED: Support for both target_column and target_col for backward compatibility
        self.target_col = getattr(self.config, 'target_column', None)
        if self.target_col is None:
            self.target_col = getattr(self.config, 'target_col', 'subscribed')
        self.model_type = getattr(self.config, 'model_type', 'auto')  # Changed to 'auto'
        self.test_size = getattr(self.config, 'test_size', 0.2)
        self.random_state = getattr(self.config, 'random_state', 42)
        self.metrics = getattr(self.config, 'metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.cross_validate = getattr(self.config, 'cross_validate', False)
        self.cv_folds = getattr(self.config, 'cv_folds', 5)
        self.class_weights = getattr(self.config, 'class_weights', None)
        self.early_stopping = getattr(self.config, 'early_stopping', False)
        self.output_dir = getattr(self.config, 'output_dir', 'model_report')
        
        #  Add proper fallback for data source
        self.df = self._get_data_from_state()
        
        #  Handle schema access with multiple fallback options
        self.schema = self._get_schema_from_state()
        
        # ADDED: Get target column from state if available (from preprocessor)
        if hasattr(self.state, 'target_column') and self.state.target_column:
            self.target_col = self.state.target_column
            self.logger.info(f"Using target column from state: {self.target_col}")
        
        # Initialize other attributes
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None
        self.results = {}
        self.feature_importance = None
        self.cv_scores = None
        self.problem_type = None  # Will be set to 'classification' or 'regression'
        
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.report_path = os.path.join(self.output_dir, "model_report.md")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def _get_data_from_state(self):
        """Get data from state with proper fallback mechanism."""
        #  Proper fallback logic
        if hasattr(self.state, 'processed_df') and self.state.processed_df is not None:
            self.logger.info("Using processed data from state")
            return self.state.processed_df
        elif hasattr(self.state, 'raw_df') and self.state.raw_df is not None:
            self.logger.info("Using raw data from state (fallback)")
            return self.state.raw_df
        else:
            self.logger.warning("No data available in state")
            return None

    def _get_schema_from_state(self):
        """Safely get schema from state with multiple fallback options."""
        if hasattr(self.state, 'schema') and self.state.schema is not None:
            self.logger.info("Using schema from state")
            return self.state.schema
        elif hasattr(self.state, 'column_types') and self.state.column_types is not None:
            # Create schema from column_types if available
            self.logger.info("Creating schema from column_types in state")
            return self._create_schema_from_column_types()
        else:
            self.logger.warning("No schema found in state - will infer from data if needed")
            return None

    def _create_schema_from_column_types(self):
        """Create schema dictionary from column_types."""
        schema = {
            'categorical_columns': [],
            'numerical_columns': [],
            'datetime_columns': [],
            'text_columns': []
        }
        
        if hasattr(self.state, 'column_types'):
            for col, col_type in self.state.column_types.items():
                if col_type == 'categorical':
                    schema['categorical_columns'].append(col)
                elif col_type == 'numerical':
                    schema['numerical_columns'].append(col)
                elif col_type == 'datetime':
                    schema['datetime_columns'].append(col)
                elif col_type == 'text':
                    schema['text_columns'].append(col)
        
        self.logger.info(f"Created schema from column_types: {schema}")
        return schema

    def _infer_schema_from_data(self, df):
        """Infer schema from DataFrame when not provided."""
        self.logger.info("Inferring schema from data...")
        
        schema = {
            'categorical_columns': [],
            'numerical_columns': [],
            'datetime_columns': [],
            'text_columns': []
        }
        
        for col in df.columns:
            if col == self.target_col:
                continue  # Skip target column
                
            col_type = str(df[col].dtype)
            
            if col_type in ['object', 'category', 'bool']:
                # Check if it's actually categorical (limited unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    schema['categorical_columns'].append(col)
                else:
                    schema['text_columns'].append(col)
            elif col_type in ['int64', 'float64', 'int32', 'float32']:
                schema['numerical_columns'].append(col)
            elif 'datetime' in col_type:
                schema['datetime_columns'].append(col)
            else:
                # Default to numerical for unknown types
                schema['numerical_columns'].append(col)
        
        self.logger.info(f"Inferred schema: {schema}")
        return schema

    def _detect_problem_type(self, y):
        """Detect if the problem is classification or regression."""
        # If target is object/string type, it's classification
        if y.dtype in ['object', 'category']:
            return 'classification'
        
        # If target is numeric with few unique values relative to sample size
        unique_values = y.nunique()
        total_samples = len(y)
        
        # If less than 10% of samples are unique, treat as classification
        if unique_values / total_samples < 0.1:
            return 'classification'
        else:
            return 'regression'

    def _save_plot(self, fig, filename):
        """Saves a plot to the plots directory and returns a relative path for the report."""
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return os.path.join("plots", filename)

    def prepare_data(self):
        """Preprocesses the data and splits it into training and testing sets."""
        self.logger.info("Preparing data...")
        
        #  Check if data is available with better error message
        if self.df is None:
            if hasattr(self.state, 'raw_df') and self.state.raw_df is not None:
                self.df = self.state.raw_df
                self.logger.warning("Using raw data as fallback - processed data not available")
            else:
                raise ValueError("No data available for training. Run DataLoadingAgent first.")
        
        #  Handle missing schema by inferring from data
        if self.schema is None:
            self.logger.warning("No schema available in state. Inferring schema from data...")
            self.schema = self._infer_schema_from_data(self.df)
        
        # ADDED: Validate target column exists
        if self.target_col not in self.df.columns:
            available_cols = self.df.columns.tolist()
            raise ValueError(f"Target column '{self.target_col}' not found in data. Available columns: {available_cols}")
        
        # Drop rows with missing target values
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[self.target_col])
        if len(self.df) < initial_rows:
            self.logger.warning(f"Dropped {initial_rows - len(self.df)} rows with missing target values")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Detect problem type
        self.problem_type = self._detect_problem_type(y)
        self.logger.info(f"Detected problem type: {self.problem_type}")

        #  Always run preprocessing to ensure categorical encoding works
        self.logger.info("Running preprocessing on features...")
        
        # Create config and state for DataPreprocessor
        class PreprocessorConfig:
            def __init__(self):
                self.name = 'data_preprocessor'
                self.handle_missing = True
                self.scale_features = True
                self.encode_categorical = True
                self.remove_outliers = True
        
        class PreprocessorState:
            def __init__(self, schema):
                self.schema = schema
                self.raw_df = None
                self.processed_df = None
        
        config = PreprocessorConfig()
        state = PreprocessorState(schema=self.schema)
        state.raw_df = X
        
        # Use the DataPreprocessor
        self.preprocessor = DataPreprocessor(config, state)
        self.preprocessor.fit(X)
        X_processed = self.preprocessor.transform(X)
        
        #  Debug check - verify preprocessing worked
        self.logger.info(f"Processed data shape: {X_processed.shape}")
        self.logger.info(f"Processed data types:\n{X_processed.dtypes}")
        
        #  Check for any remaining string columns that could cause issues
        string_columns = X_processed.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            self.logger.warning(f"String columns found after preprocessing: {list(string_columns)}")
            # Convert any remaining string columns to numeric
            for col in string_columns:
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                    X_processed[col] = X_processed[col].fillna(0)
                    self.logger.info(f"Converted column '{col}' to numeric")
                except Exception as e:
                    self.logger.error(f"Could not convert column '{col}' to numeric: {e}")
                    # Drop problematic columns
                    X_processed = X_processed.drop(columns=[col])
                    self.logger.warning(f"Dropped column '{col}' due to conversion issues")

        # Encode target variable for classification
        if self.problem_type == 'classification':
            self.logger.info(f"Original target values: {y.unique()}")
            y_encoded = self._encode_target(y)
            self.logger.info(f"Encoded target values: {np.unique(y_encoded)}")
        else:
            # For regression, keep target as is
            y_encoded = y.values
            self.logger.info(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Split data - use stratification only for classification
        if self.problem_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y_encoded, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=y_encoded
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y_encoded, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
        
        self.logger.info(f"Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        self.logger.info("Data preparation complete.")

    def _encode_target(self, y):
        """Encode target variable to numeric values."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            positive_indicators = ['yes', 'true', '1', 'positive', 'y', 't']
            negative_indicators = ['no', 'false', '0', 'negative', 'n', 'f']
            
            y_lower = y.astype(str).str.lower().str.strip()
            
            positive_mask = y_lower.isin(positive_indicators)
            negative_mask = y_lower.isin(negative_indicators)
            
            if positive_mask.any() and negative_mask.any():
                return np.where(positive_mask, 1, 0)
            else:
                le = LabelEncoder()
                return le.fit_transform(y)
        else:
            return y.values

    def train_model(self):
        """Initializes and trains the selected machine learning model."""
        # Auto-select model based on problem type if model_type is 'auto'
        if self.model_type == 'auto':
            if self.problem_type == 'classification':
                self.model_type = 'logistic'
            else:
                self.model_type = 'linear'
        
        self.logger.info(f"Training {self.model_type.replace('_', ' ').title()} model for {self.problem_type}...")
        
        # Model selection based on problem type
        if self.problem_type == 'classification':
            if self.model_type == 'logistic':
                self.model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight=self.class_weights
                )
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    class_weight=self.class_weights
                )
            else:
                raise ValueError(f"Unknown classification model type: {self.model_type}")
        else:  # regression
            if self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'ridge':
                self.model = Ridge(random_state=self.random_state)
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    random_state=self.random_state,
                    n_estimators=100
                )
            else:
                raise ValueError(f"Unknown regression model type: {self.model_type}")
        
        # Cross-validation if enabled
        if self.cross_validate:
            try:
                self.logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
                scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
                cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                          cv=self.cv_folds, scoring=scoring)
                self.cv_scores = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'all_scores': cv_scores.tolist()
                }
                self.logger.info(f"CV {scoring.title()}: {self.cv_scores['mean_score']:.3f} ± {self.cv_scores['std_score']:.3f}")
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}. Continuing without CV.")
                self.cross_validate = False
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Model training complete.")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns.tolist(),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            if self.problem_type == 'classification':
                importance_values = abs(self.model.coef_[0])
            else:
                importance_values = abs(self.model.coef_)
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns.tolist(),
                'importance': importance_values
            }).sort_values('importance', ascending=False)

    def evaluate_model(self):
        """Evaluates the model on the test set and generates metrics and plots."""
        self.logger.info("Evaluating model...")
        
        y_pred = self.model.predict(self.X_test)
        
        if self.problem_type == 'classification':
            # Classification metrics
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

            metrics_to_calculate = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics_to_calculate['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store only requested metrics
            for metric in self.metrics:
                if metric in metrics_to_calculate:
                    self.results[metric] = metrics_to_calculate[metric]

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            class_labels = [f'Class {i}' for i in range(len(np.unique(self.y_test)))]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_labels, yticklabels=class_labels)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14)
            self.results['confusion_matrix_path'] = self._save_plot(fig, 'confusion_matrix.png')

            # ROC Curve (only for binary classification with probabilities)
            if y_pred_proba is not None and 'roc_auc' in self.results and len(np.unique(self.y_test)) == 2:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {self.results["roc_auc"]:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('ROC Curve', fontsize=14)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                self.results['roc_curve_path'] = self._save_plot(fig, 'roc_curve.png')
                
        else:
            # Regression metrics
            metrics_to_calculate = {
                'mse': mean_squared_error(self.y_test, y_pred),
                'mae': mean_absolute_error(self.y_test, y_pred),
                'r2': r2_score(self.y_test, y_pred)
            }
            
            # Store metrics
            for metric in ['mse', 'mae', 'r2']:
                if metric in metrics_to_calculate:
                    self.results[metric] = metrics_to_calculate[metric]
            
            # Scatter plot for regression
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.y_test, y_pred, alpha=0.6)
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title('Actual vs Predicted Values', fontsize=14)
            self.results['scatter_plot_path'] = self._save_plot(fig, 'scatter_plot.png')

        # Feature Importance
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_features = self.feature_importance.head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            title_suffix = 'Classification' if self.problem_type == 'classification' else 'Regression'
            ax.set_title(f'Top 10 Feature Importances ({title_suffix})', fontsize=14)
            self.results['feature_importance_path'] = self._save_plot(fig, 'feature_importance.png')
        
        self.logger.info("Model evaluation complete.")

    def generate_report(self):
        """Generates a professional Markdown report summarizing the entire process."""
        self.logger.info("Generating final report...")
        
        with open(self.report_path, 'w') as f:
            f.write("# AI Model Training Report\n\n")
            
            f.write("## 1. Introduction\n")
            f.write(f"This report details the training and evaluation of a **{self.model_type.replace('_', ' ').title()}** model for **{self.problem_type}**. ")
            f.write(f"The primary objective was to predict the target variable **`{self.target_col}`**.\n\n")

            f.write("## 2. Model Configuration\n")
            f.write("| Parameter | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| **Problem Type** | {self.problem_type} |\n")
            f.write(f"| **Model Type** | {self.model_type} |\n")
            f.write(f"| **Target Variable** | {self.target_col} |\n")
            f.write(f"| **Test Size** | {self.test_size} |\n")
            f.write(f"| **Random State** | {self.random_state} |\n")
            if self.problem_type == 'classification':
                f.write(f"| **Class Weights** | {self.class_weights} |\n")
            f.write(f"| **Cross Validation** | {self.cross_validate} |\n")
            if self.cross_validate:
                f.write(f"| **CV Folds** | {self.cv_folds} |\n")
            f.write(f"| **Training Samples** | {len(self.X_train)} |\n")
            f.write(f"| **Test Samples** | {len(self.X_test)} |\n\n")

            f.write("## 3. Results\n\n")
            
            f.write("### 3.1. Evaluation Metrics\n")
            f.write("| Metric | Score |\n")
            f.write("| :--- | :--- |\n")
            for metric, value in self.results.items():
                if metric not in ['confusion_matrix_path', 'roc_curve_path', 'scatter_plot_path', 'feature_importance_path']:
                    f.write(f"| **{metric.upper()}** | {value:.3f} |\n")
            f.write("\n")

            if self.cross_validate and self.cv_scores:
                f.write("### 3.2. Cross-Validation Results\n")
                scoring_metric = 'Accuracy' if self.problem_type == 'classification' else 'R²'
                f.write(f"- Mean {scoring_metric}: {self.cv_scores['mean_score']:.3f}\n")
                f.write(f"- Std Deviation: {self.cv_scores['std_score']:.3f}\n")
                f.write(f"- Fold Scores: {self.cv_scores['all_scores']}\n\n")

            f.write("### 3.3. Visualizations\n")
            if 'confusion_matrix_path' in self.results:
                f.write("#### Confusion Matrix\n")
                f.write(f"![Confusion Matrix]({self.results['confusion_matrix_path']})\n\n")
            
            if 'roc_curve_path' in self.results:
                f.write("#### ROC Curve\n")
                f.write(f"![ROC Curve]({self.results['roc_curve_path']})\n\n")
            
            if 'scatter_plot_path' in self.results:
                f.write("#### Actual vs Predicted Values\n")
                f.write(f"![Scatter Plot]({self.results['scatter_plot_path']})\n\n")
            
            if 'feature_importance_path' in self.results:
                f.write("#### Feature Importance\n")
                f.write(f"![Feature Importance]({self.results['feature_importance_path']})\n\n")

            f.write("## 4. Conclusion\n")
            if self.problem_type == 'classification':
                accuracy = self.results.get('accuracy', 0)
                if accuracy > 0.8:
                    f.write("The model demonstrates excellent performance with high accuracy.\n\n")
                elif accuracy > 0.7:
                    f.write("The model shows good performance with potential for improvement.\n\n")
                else:
                    f.write("The model performance requires further optimization.\n\n")
            else:
                r2 = self.results.get('r2', 0)
                if r2 > 0.7:
                    f.write("The model demonstrates excellent performance with high R² score.\n\n")
                elif r2 > 0.5:
                    f.write("The model shows good performance with reasonable explanatory power.\n\n")
                else:
                    f.write("The model performance requires further optimization.\n\n")

            f.write("**Recommendations**:\n")
            f.write("- Experiment with different algorithms\n")
            f.write("- Perform hyperparameter tuning\n")
            f.write("- Collect more training data\n")
            f.write("- Engineer additional features\n")

    def run(self) -> str:
        """Runs the full training and evaluation pipeline."""
        self.logger.info("Starting model training pipeline...")
        
        try:
            self.prepare_data()
            self.train_model()
            self.evaluate_model()
            self.generate_report()
            
            message = f"Model training complete. Report saved to: {self.report_path}"
            self.logger.info(message)
            return message
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

if __name__ == '__main__':
    # Sample DataFrame for testing
    data = {
        'age': [25, 30, 45, 35, 50, 22, 60, 28, 40, 33, 29, 48, 38, 55, 41, 65, 23, 31, 39, 52],
        'salary': [50, 60, 80, 75, 90, 48, 120, 52, 85, 72, 62, 88, 78, 110, 92, 130, 51, 65, 81, 95],
        'city': ['New York', 'London', 'Tokyo', 'London', 'New York', 'Tokyo', 'New York', 'Tokyo', 'New York', 'London', 
                'Tokyo', 'New York', 'London', 'New York', 'Tokyo', 'London', 'Tokyo', 'New York', 'London', 'New York'],
        'experience': [2, 5, 10, 8, 15, 1, 20, 3, 12, 6, 4, 14, 9, 18, 11, 25, 1, 6, 10, 16],
        'subscribed': ['no', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 
                      'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
    }
    sample_df = pd.DataFrame(data)
    
    # Create config with all required parameters
    class TrainerConfig:
        def __init__(self):
            self.name = 'model_trainer'
            self.target_col = 'subscribed'
            self.model_type = 'auto'  # Changed to auto
            self.test_size = 0.2
            self.random_state = 42
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.cross_validate = False  #  Disabled for testing
            self.cv_folds = 5
            self.class_weights = 'balanced'
            self.output_dir = 'model_report'
    
    # Get schema using standalone function
    schema = inspect_data(sample_df, categorical_threshold=0.5)
    
    # FIXED: Create state with all required attributes
    class TrainerState:
        def __init__(self, df, schema):
            self.raw_df = df
            self.processed_df = None  # ADDED: processed_df attribute
            self.schema = schema
            self.target_column = 'subscribed'  # ADDED: target_column attribute
    
    # Initialize and run trainer
    config = TrainerConfig()
    state = TrainerState(sample_df, schema)
    
    print("--- Running Model Trainer ---")
    trainer = ModelTrainer(config, state)
    result = trainer.run()
    print(result)
