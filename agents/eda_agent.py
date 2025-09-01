import sys
import os
sys.path.append('/Users/apple/Desktop/multi_agents/agents')

from base_agent import BaseAgent, AgentConfig, PipelineState 
from exploratory_data_analyzer import EDAAnalyzer  #  Correct import

class EDAAgent(BaseAgent):
    """An agent responsible for performing exploratory data analysis."""

    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)
        # Config se parameters lo
        self.target_col = getattr(self.config, 'target_column', None)
        # ADDED: Fallback to target_col for backward compatibility
        if self.target_col is None:
            self.target_col = getattr(self.config, 'target_col', None)
        self.ordinal_cols = getattr(self.config, 'ordinal_columns', [])
        self.generate_plots = getattr(self.config, 'generate_plots', True)
        self.output_format = getattr(self.config, 'output_format', 'md')
        self.plot_style = getattr(self.config, 'plot_style', 'seaborn')

    def run(self) -> str:  #  Synchronous method
        """
        Runs the EDAAnalyzer on the raw data and generates a report.
        """
        self.logger.info("Performing Exploratory Data Analysis...")
        
        if self.state.raw_df is None:
            raise ValueError("Raw DataFrame not found. Run DataLoadingAgent first.")
        
        if self.state.schema is None:
            raise ValueError("Schema not found. Run ColumnTypeDetector first.")

        # Use processed_df if available, else raw_df
        data_for_eda = self.state.processed_df if self.state.processed_df is not None else self.state.raw_df

        # ADDED: Get target column from state if available (from preprocessor)
        target_column = self.target_col
        if target_column is None and hasattr(self.state, 'target_column'):
            target_column = self.state.target_column
            self.logger.info(f"Using target column from state: {target_column}")

        analyzer = EDAAnalyzer(
            df=data_for_eda,
            schema=self.state.schema,
            target_col=target_column,  # UPDATED: Use the resolved target column
            ordinal_cols=self.ordinal_cols,
            generate_plots=self.generate_plots,
            output_format=self.output_format,
            plot_style=self.plot_style
        )
        
        analyzer.run_analysis()
        self.state.eda_report_path = analyzer.report_path
        
        message = f"EDA report generated at {self.state.eda_report_path}"
        self.logger.info(message)
        return message
