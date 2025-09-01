# agents/base_agent.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import time

@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    timeout: int = 60
    retry_attempts: int = 3
    enabled: bool = True

    # Add all possible configuration parameters for different agents
    supported_formats: List[str] = None
    encoding: str = 'utf-8'

    #### data inspection parameter 
    sample_size: int = 1000
    generate_summary: bool = True
    detect_column_types: bool = True
    categorical_threshold: float = 0.05 
    
    ### data preprocessor parameter
    handle_missing: bool = True
    scale_features: bool = True
    encode_categorical: bool = True
    remove_outliers: bool = True
    
    ### EDA parameter 
    generate_plots: bool = True
    output_format: str = 'md'
    plot_style: str = 'seaborn'
    target_column: str = None
    ordinal_columns: List[str] = None
    openai_api_key: str = None

    # Exploratory Analysis parameters
    correlation_threshold: float = 0.8
    generate_insights: bool = True
    top_features_count: int = 10

    #  MODEL TRAINER PARAMETERS 
    target_col: str = 'subscribed'
    model_type: str = 'logistic'
    test_size: float = 0.2
    random_state: int = 42
    metrics: List[str] = None
    cross_validate: bool = False
    cv_folds: int = 5
    class_weights: str = None
    early_stopping: bool = False
    output_dir: str = 'model_report'

    # Modeling parameters
    test_size: float = 0.2
    models: List[str] = None
    metric: str = 'accuracy'
    cross_validation: bool = True
    folds: int = 5
    
    def __post_init__(self):
        """Initialize default values for lists"""
        if self.supported_formats is None:
            self.supported_formats = ['.csv', '.xlsx', '.parquet']
        if self.models is None:
            self.models = ['logistic', 'random_forest']
        if self.ordinal_columns is None:
            self.ordinal_columns = []
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

@dataclass
class PipelineState:
    """A dataclass to hold the shared state between agents in the pipeline."""
    raw_df: pd.DataFrame = None
    processed_df: pd.DataFrame = None
    target_series: pd.Series = None
    schema: Dict[str, str] = None
    eda_report_path: str = None
    model_report_path: str = None
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    file_path: str = None
    #  Add model training results to state
    model_training_results: Dict[str, Any] = field(default_factory=dict)
    modeling_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """The output of a single agent's execution."""
    success: bool
    agent_name: str
    message: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None
    output_data: Optional[Any] = None
    #  Add additional metadata
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

# --- Base Agent (Synchronous Version) ---

class BaseAgent:
    """
    Base class for all agents in the data analysis orchestrator.
    """
    def __init__(self, config: AgentConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger for the agent."""
        logger = logging.getLogger(self.config.name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def check_enabled(self) -> bool:
        """Check if the agent is enabled."""
        if not self.config.enabled:
            self.logger.warning(f"Agent {self.config.name} is disabled. Skipping execution.")
            return False
        return True

    def validate_config(self) -> bool:
        """Validate agent configuration."""
        # Basic validation - can be overridden by subclasses
        if not self.config.name:
            self.logger.error("Agent name is required")
            return False
        return True

    def run(self) -> str:
        """
        The main entry point for an agent's execution.
        Returns:
            A string message summarizing the result.
        """
        raise NotImplementedError("Subclasses must implement run() method")

    def execute(self) -> AnalysisResult:
        """
        A wrapper method that handles execution with logging, and timing.
        """
        # Check if agent is enabled
        if not self.check_enabled():
            return AnalysisResult(
                success=True,
                agent_name=self.config.name,
                message="Agent disabled - skipped execution",
                execution_time=0.0
            )
        
        # Validate configuration
        if not self.validate_config():
            return AnalysisResult(
                success=False,
                agent_name=self.config.name,
                error="Invalid configuration"
            )
        
        self.logger.info("Starting execution...")
        start_time = time.time()
        
        try:
            result_message = self.run()
            execution_time = time.time() - start_time
            
            self.logger.info(f"Execution successful in {execution_time:.2f} seconds.")
            
            # Create result with additional metadata
            result = AnalysisResult(
                success=True,
                agent_name=self.config.name,
                message=result_message,
                execution_time=execution_time,
                output_data=getattr(self, 'output', None)
            )
            
            # Add metrics and artifacts if available
            if hasattr(self, 'metrics'):
                result.metrics = getattr(self, 'metrics', {})
            if hasattr(self, 'artifacts'):
                result.artifacts = getattr(self, 'artifacts', {})
                
            return result
            
        except TimeoutError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Execution timed out after {execution_time:.2f} seconds: {e}")
            return AnalysisResult(
                success=False,
                agent_name=self.config.name,
                error=f"Execution timed out: {e}",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"An unexpected error occurred after {execution_time:.2f} seconds: {e}", exc_info=True)
            return AnalysisResult(
                success=False,
                agent_name=self.config.name,
                error=f"An unexpected error occurred: {e}",
                execution_time=execution_time
            )

    def save_artifact(self, key: str, value: Any):
        """Save an artifact to the state."""
        if not hasattr(self.state, 'agent_outputs'):
            self.state.agent_outputs = {}
        self.state.agent_outputs[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Retrieve an artifact from the state."""
        return getattr(self.state, 'agent_outputs', {}).get(key, default)

#  Optional: Add utility functions for common operations

def create_agent_config(**kwargs) -> AgentConfig:
    """Helper function to create AgentConfig with given parameters."""
    return AgentConfig(**kwargs)

def create_pipeline_state(**kwargs) -> PipelineState:
    """Helper function to create PipelineState with given parameters."""
    return PipelineState(**kwargs)
