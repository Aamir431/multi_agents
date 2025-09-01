from .agents.base_agent import BaseAgent
from logic.data_schema_detector import detect_column_types

class SchemaDetectionAgent(BaseAgent):
    """An agent responsible for detecting the schema of the raw DataFrame."""

    async def run(self) -> str:
        """
        Analyzes the raw DataFrame to detect column types and updates the shared state.
        """
        self.logger.info("Detecting data schema...")
        if self.state.raw_df is None:
            raise ValueError("Raw DataFrame not found in pipeline state. The DataLoadingAgent must run first.")
            
        schema = detect_column_types(self.state.raw_df)
        self.state.schema = schema
        
        message = f"Schema detected for {len(schema)} columns."
        self.logger.info(message)
        return message
