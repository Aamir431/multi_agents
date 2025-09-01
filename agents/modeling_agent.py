import sys
sys.path.append('/Users/apple/Desktop/multi_agents/agents')

from base_agent import BaseAgent, AgentConfig, PipelineState
from model_trainer import ModelTrainer

class ModelingAgent(BaseAgent):
    """An agent responsible for training and evaluating a machine learning model."""

    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)
        #  Backward compatibility for config params
        self.target_col = getattr(self.config, 'target_column', None)
        if self.target_col is None:
            self.target_col = getattr(self.config, 'target_col', 'subscribed')
        
        #  Default model_type aligned with trainer ("auto" supported)
        self.model_type = getattr(self.config, 'model_type', 'auto')
        self.test_size = getattr(self.config, 'test_size', 0.2)
        self.random_state = getattr(self.config, 'random_state', 42)
        
        #  Ensure full metric set (includes ROC AUC)
        self.metrics = getattr(
            self.config,
            'metrics',
            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )
        self.cross_validate = getattr(self.config, 'cross_validate', False)
        self.cv_folds = getattr(self.config, 'cv_folds', 5)

    def run(self):
        """
        Runs the ModelTrainer on the processed data and generates a report.
        """
        self.logger.info(f"Training and evaluating model (type={self.model_type})...")
        
        #  If preprocessing agent stored target_column in state, prefer that
        if hasattr(self.state, 'target_column') and self.state.target_column:
            self.target_col = self.state.target_column
            self.logger.info(f"Using target column from state: {self.target_col}")
        
        if self.state.processed_df is None:
            raise ValueError("Processed data not found. Preprocessing must run first.")
        
        # Check target column presence
        if self.target_col not in self.state.processed_df.columns:
            available_cols = list(self.state.processed_df.columns)
            error_msg = (
                f"Target column '{self.target_col}' not found in processed data. "
                f"Available columns: {available_cols}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        #  Define local config class for ModelTrainer
        class ModelTrainerConfig:
            def __init__(self, target_col, model_type, test_size, random_state,
                         metrics, cross_validate, cv_folds):
                self.name = 'model_trainer'
                self.target_column = target_col   # for consistency
                self.target_col = target_col      # backward compatibility
                self.model_type = model_type
                self.test_size = test_size
                self.random_state = random_state
                self.metrics = metrics
                self.cross_validate = cross_validate
                self.cv_folds = cv_folds
                self.class_weights = 'balanced'
                self.output_dir = 'model_report'
        
        # Create trainer config
        trainer_config = ModelTrainerConfig(
            target_col=self.target_col,
            model_type=self.model_type,
            test_size=self.test_size,
            random_state=self.random_state,
            metrics=self.metrics,
            cross_validate=self.cross_validate,
            cv_folds=self.cv_folds
        )
        
        # Initialize and run ModelTrainer
        trainer = ModelTrainer(trainer_config, self.state)
        
        try:
            result_message = trainer.run()
            
            # Store results in pipeline state
            self.state.model_report_path = trainer.report_path
            self.state.model_metrics = trainer.results
            
            message = f"Model training completed: {result_message}"
            self.logger.info(message)
            
            return {
                'success': True,
                'message': message,
                'report_path': self.state.model_report_path,
                'metrics': trainer.results,
                'agent_name': 'ModelingAgent'
            }
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'agent_name': 'ModelingAgent'
            }

#  Backward compatibility alias
def execute(self):
    """Alias for run() to maintain compatibility with orchestrator."""
    return self.run()

ModelingAgent.execute = execute
