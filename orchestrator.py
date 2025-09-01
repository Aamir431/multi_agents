import yaml
import os
import sys
import pandas as pd

# Add agents directory to path to find local modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

try:
    from logger import setup_logger
except ImportError:
    # Fallback if logger is not available
    import logging
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        return logger

# Import base classes from agents directory
try:
    from agents.base_agent import AgentConfig, PipelineState
except ImportError:
    # Try alternative import path
    try:
        from agents.base_agent import AgentConfig, PipelineState
    except ImportError as e:
        print(f"Error importing base_agent: {e}")
        print("Make sure base_agent.py exists in agents/ directory")
        raise


class PipelineOrchestrator:
    def __init__(self, config_path='config.yaml', target_variable=None):  # ADDED target_variable parameter
        self.logger = setup_logger('Orchestrator')
        self.config = self.load_config(config_path)
        self.target_variable = target_variable  # ADDED: Store target variable
        self.agents = {}
        self.shared_state = PipelineState()  # Create shared state for all agents

    # ---------- helpers to handle dict OR object agent results ----------
    @staticmethod
    def _is_success(result, default=False):
        if isinstance(result, dict):
            return bool(result.get('success', default))
        return bool(getattr(result, 'success', default))

    @staticmethod
    def _get_field(result, key, default=None):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    @staticmethod
    def _ensure_dict(result):
        if isinstance(result, dict):
            return result
        # Try to convert simple objects to dict
        if hasattr(result, '__dict__'):
            return dict(result.__dict__)
        # Fallback minimal shape
        return {'success': bool(result)}

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Config file {config_path} not found")
            # Create default config if not exists
            default_config = {
                'data_loading': {
                    'supported_formats': ['.csv'],
                    'encoding': 'utf-8',
                    'timeout': 60,
                    'retry_attempts': 3
                },
                'data_inspection': {
                    'sample_size': 1000,
                    'generate_summary': True,
                    'detect_column_types': True,
                    'timeout': 45
                },
                'preprocessing': {
                    'handle_missing': True,
                    'scale_features': True,
                    'encode_categorical': True,
                    'remove_outliers': True,
                    'timeout': 90
                },
                'eda': {
                    'generate_plots': True,
                    'output_format': 'md',
                    'plot_style': 'seaborn',
                    'timeout': 120
                },
                'model_training': {
                    'target_col': 'subscribed',
                    'model_type': 'logistic',
                    'test_size': 0.2,
                    'random_state': 42,
                    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    'cross_validate': True,
                    'cv_folds': 5,
                    'class_weights': 'balanced',
                    'output_dir': 'model_report',
                    'timeout': 180,
                    'retry_attempts': 2
                },
                'modeling': {
                    'target_col': 'subscribed',  # for ModelingAgent
                    'model_type': 'logistic',
                    'test_size': 0.2,
                    'models': ['logistic'],
                    'metric': 'accuracy',
                    'cross_validation': True,
                    'folds': 5,
                    'timeout': 150
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
            self.logger.info(f"Created default config file: {config_path}")
            return default_config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            raise

    def initialize_agents(self):
        self.logger.info("Initializing agents...")
        try:
            # Import agents
            from agents.data_loading import DataLoadingAgent
            from agents.data_inspector import ColumnTypeDetector
            from agents.data_preprocessor import DataPreprocessor
            from agents.eda_agent import EDAAgent
            from agents.model_trainer import ModelTrainer
            from agents.modeling_agent import ModelingAgent

            # Get config sections
            data_loading_config = self.config.get('data_loading', {})
            data_inspection_config = self.config.get('data_inspection', {})
            preprocessing_config = self.config.get('preprocessing', {})
            eda_config = self.config.get('eda', {})
            model_training_config = self.config.get('model_training', {})
            modeling_config = self.config.get('modeling', {})

            # UPDATE CONFIG WITH TARGET VARIABLE IF PROVIDED
            if self.target_variable:
                # Update target column in all relevant config sections
                if 'preprocessing' in self.config:
                    self.config['preprocessing']['target_column'] = self.target_variable
                if 'model_training' in self.config:
                    self.config['model_training']['target_col'] = self.target_variable
                    self.config['model_training']['target_column'] = self.target_variable  # add alias
                if 'modeling' in self.config:
                    self.config['modeling']['target_col'] = self.target_variable
                    self.config['modeling']['target_column'] = self.target_variable  # add alias

                # Update the config objects that will be passed to agents
                preprocessing_config['target_column'] = self.target_variable
                model_training_config['target_col'] = self.target_variable
                model_training_config['target_column'] = self.target_variable
                modeling_config['target_col'] = self.target_variable
                modeling_config['target_column'] = self.target_variable

            # Initialize agents with both config and state
            self.agents['data_loader'] = DataLoadingAgent(
                config=AgentConfig(name="DataLoader", **data_loading_config),
                state=self.shared_state
            )

            self.agents['inspector'] = ColumnTypeDetector(
                config=AgentConfig(name="ColumnTypeDetector", **data_inspection_config),
                state=self.shared_state
            )

            self.agents['preprocessor'] = DataPreprocessor(
                config=AgentConfig(name="DataPreprocessor", **preprocessing_config),
                state=self.shared_state
            )

            self.agents['eda_agent'] = EDAAgent(
                config=AgentConfig(name="EDAAgent", **eda_config),
                state=self.shared_state
            )

            # ModelTrainer agent
            self.agents['model_trainer'] = ModelTrainer(
                config=AgentConfig(name="ModelTrainer", **model_training_config),
                state=self.shared_state
            )

            # ModelingAgent
            self.agents['modeling_agent'] = ModelingAgent(
                config=AgentConfig(name="ModelingAgent", **modeling_config),
                state=self.shared_state
            )

            self.logger.info("All agents initialized successfully!")
            return True

        except ImportError as e:
            self.logger.error(f"Failed to import agent: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            return False

    def run_pipeline(self, data_path):
        self.logger.info(f"Starting pipeline for: {data_path}")

        # ADDED: Log target variable if provided
        if self.target_variable:
            self.logger.info(f"Target variable: {self.target_variable}")

        try:
            # Set file path in shared state for data loader
            self.shared_state.file_path = data_path

            # Step 1: Load data
            self.logger.info("Step 1: Loading data...")
            result = self.agents['data_loader'].execute()
            if not self._is_success(result):
                raise Exception(f"Data loading failed: {self._get_field(result, 'error', 'Unknown error')}")

            data = self.shared_state.raw_df
            self.logger.info(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")

            # ADDED: Validate target variable exists in data
            if self.target_variable and self.target_variable not in data.columns:
                available_columns = data.columns.tolist()
                raise ValueError(
                    f"Target variable '{self.target_variable}' not found in data. "
                    f"Available columns: {available_columns}"
                )

            # Step 2: Inspect data
            self.logger.info("Step 2: Inspecting data...")
            inspection_result = self.agents['inspector'].execute()
            if not self._is_success(inspection_result):
                raise Exception(f"Data inspection failed: {self._get_field(inspection_result, 'error', 'Unknown error')}")

            # Ensure schema is in state
            if hasattr(self.shared_state, 'schema') and self.shared_state.schema:
                self.logger.info(f"Schema found in state: {len(self.shared_state.schema)} column types detected")
            elif hasattr(self.shared_state, 'column_types') and self.shared_state.column_types:
                self.logger.info(f"Column types found in state: {len(self.shared_state.column_types)} columns typed")
                self.shared_state.schema = self._create_schema_from_column_types(self.shared_state.column_types)
            else:
                self.logger.warning("No schema or column_types found in state after inspection")

            # Step 3: Preprocess data
            self.logger.info("Step 3: Preprocessing data...")
            preprocessing_result = self.agents['preprocessor'].execute()
            if not self._is_success(preprocessing_result):
                raise Exception(f"Data preprocessing failed: {self._get_field(preprocessing_result, 'error', 'Unknown error')}")

            processed_data = self.shared_state.processed_df
            self.logger.info(f"Data preprocessing completed: {len(processed_data)} rows")

            # Step 4: Generate EDA report
            self.logger.info("Step 4: Generating EDA report...")
            eda_result = self.agents['eda_agent'].execute()
            if not self._is_success(eda_result):
                self.logger.warning(f"EDA had issues: {self._get_field(eda_result, 'error', 'Unknown error')}")
            else:
                self.logger.info("EDA report generated")

            # DEBUG: Check what's in state before ModelTrainer
            self._log_state_contents("Before ModelTrainer")

            # Step 5: Train and evaluate model using ModelTrainer
            self.logger.info("Step 5: Training and evaluating model...")
            model_training_result = self.agents['model_trainer'].execute()
            if not self._is_success(model_training_result):
                raise Exception(f"Model training failed: {self._get_field(model_training_result, 'error', 'Unknown error')}")

            self.logger.info("Model training and evaluation completed")

            # Step 6: Train models with ModelingAgent (optional step)
            self.logger.info("Step 6: Training additional models...")
            modeling_result = self.agents['modeling_agent'].execute()
            if not self._is_success(modeling_result):
                self.logger.warning(f"Modeling agent had issues: {self._get_field(modeling_result, 'error', 'Unknown error')}")
            else:
                self.logger.info("Additional models trained successfully")

            self.logger.info("🎉 Pipeline completed successfully!")
            return {
                'data': processed_data,
                'inspection_report': self._ensure_dict(inspection_result),
                'eda_report': self._ensure_dict(eda_result),
                'model_training_result': self._ensure_dict(model_training_result),
                'modeling_result': self._ensure_dict(modeling_result)
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            # Save error to state for debugging
            self.shared_state.agent_outputs['pipeline_error'] = str(e)
            raise

    def _create_schema_from_column_types(self, column_types):
        """Create schema dictionary from column_types."""
        schema = {
            'categorical_columns': [],
            'numerical_columns': [],
            'datetime_columns': [],
            'text_columns': []
        }

        for col, col_type in column_types.items():
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

    def _log_state_contents(self, stage):
        """Log what's in the state for debugging."""
        self.logger.info(f"State contents at {stage}:")
        for attr in dir(self.shared_state):
            if not attr.startswith('_') and not callable(getattr(self.shared_state, attr)):
                value = getattr(self.shared_state, attr)
                if value is not None:
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        self.logger.info(f"  {attr}: {type(value).__name__} shape {value.shape}")
                    elif isinstance(value, dict):
                        self.logger.info(f"  {attr}: dict with {len(value)} keys")
                    else:
                        self.logger.info(f"  {attr}: {type(value).__name__}")

    def get_agent_status(self):
        """Returns the status of all initialized agents"""
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                'initialized': agent is not None,
                'config': agent.config.__dict__ if agent else {}
            }
        return status


# Simple executor - MODIFIED to accept target_variable
def run_simple_pipeline(data_path, config_path='config.yaml', target_variable=None):  # ADDED target_variable
    """
    Simple function to run the complete pipeline
    """
    orchestrator = PipelineOrchestrator(config_path, target_variable)  # PASS target_variable
    if orchestrator.initialize_agents():
        return orchestrator.run_pipeline(data_path)
    else:
        raise Exception("Failed to initialize agents")


# ✅ Fixed agent keys & robust success checks
def run_pipeline_step_by_step(data_path, config_path='config.yaml', target_variable=None):  # ADDED target_variable
    """
    Run pipeline with step-by-step control
    """
    orchestrator = PipelineOrchestrator(config_path, target_variable)  # PASS target_variable
    if not orchestrator.initialize_agents():
        raise Exception("Failed to initialize agents")

    def _ok(res):  # local helper to tolerate dict/object
        if isinstance(res, dict):
            return bool(res.get('success', False))
        return bool(getattr(res, 'success', False))

    results = {}
    steps = [
        ('data_loader', "Loading data"),
        ('inspector', "Inspecting data"),
        ('preprocessor', "Preprocessing data"),
        ('eda_agent', "Generating EDA report"),
        ('model_trainer', "Training model"),
        ('modeling_agent', "Training additional models")
    ]

    for agent_key, step_description in steps:
        try:
            orchestrator.logger.info(f"Starting {step_description}...")
            res = orchestrator.agents[agent_key].execute()
            results[agent_key] = orchestrator._ensure_dict(res)
            if not _ok(res):
                orchestrator.logger.error(f"{step_description} failed: {orchestrator._get_field(res, 'error', 'Unknown error')}")
                break
            orchestrator.logger.info(f"{step_description} completed successfully")
        except Exception as e:
            orchestrator.logger.error(f"Error in {step_description}: {e}")
            results[agent_key] = {
                'success': False,
                'error': str(e),
                'agent_name': agent_key
            }
            break

    return results


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        config_file = sys.argv[2] if len(sys.argv) > 2 else 'config.yaml'
        target_var = sys.argv[3] if len(sys.argv) > 3 else None  # ADDED: Optional target variable

        try:
            results = run_simple_pipeline(data_file, config_file, target_var)  # PASS target_var
            print("Pipeline completed successfully!")
            print(f"Model report saved to: {results['model_training_result'].get('report_path', 'Unknown')}")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python orchestrator.py <data_file> [config_file] [target_variable]")
        print("Example: python orchestrator.py data.csv config.yaml Sales")
        print("Example: python orchestrator.py data.csv config.yaml")
