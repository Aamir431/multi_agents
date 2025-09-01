# agents/data_loading.py
import pandas as pd
import chardet 
from base_agent import BaseAgent, AgentConfig, PipelineState

class DataLoadingAgent(BaseAgent):
    """An agent responsible for loading the initial dataset from a file."""

    def __init__(self, config: AgentConfig, state: PipelineState):
        super().__init__(config, state)

    def detect_file_encoding(self, file_path):
        """
        Automatically detect the file encoding using chardet.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 10KB to detect encoding
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                
                if result['confidence'] > 0.7:  # Only use if confident
                    self.logger.info(f"Detected encoding: {result['encoding']} (confidence: {result['confidence']:.2f})")
                    return result['encoding']
                else:
                    self.logger.warning(f"Low confidence encoding detection: {result['encoding']} (confidence: {result['confidence']:.2f})")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}")
            return None

    def try_multiple_encodings(self, file_path):
        """
        Try reading the file with multiple common encodings.
        """
        common_encodings = [
            'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252',
            'utf-16', 'ascii', 'mac_roman'
        ]
        
        for encoding in common_encodings:
            try:
                self.logger.info(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully read with {encoding} encoding")
                return df, encoding
            except (UnicodeDecodeError, LookupError) as e:
                self.logger.debug(f"Failed with {encoding}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Unexpected error with {encoding}: {e}")
                continue
        
        return None, None

    def run(self) -> str:
        """
        Loads data from the specified CSV file into the shared pipeline state.
        """
        # Get file_path from state (set by orchestrator)
        file_path = self.state.file_path
        if not file_path:
            raise ValueError("File path not provided in state")
        
        self.logger.info(f"Loading data from {file_path}...")
        
        # Check file format using config
        if self.config.supported_formats and not any(file_path.endswith(fmt) for fmt in self.config.supported_formats):
            raise ValueError(f"Invalid file type. Supported formats: {self.config.supported_formats}")
        
        # Try configured encoding first
        configured_encoding = getattr(self.config, 'encoding', 'utf-8')
        self.logger.info(f"Trying configured encoding: {configured_encoding}")
        
        try:
            df = pd.read_csv(file_path, encoding=configured_encoding)
            used_encoding = configured_encoding
            
        except (UnicodeDecodeError, LookupError) as e:
            self.logger.warning(f"Configured encoding '{configured_encoding}' failed: {e}")
            
            # Step 1: Try automatic encoding detection
            detected_encoding = self.detect_file_encoding(file_path)
            if detected_encoding and detected_encoding != configured_encoding:
                try:
                    self.logger.info(f"Trying detected encoding: {detected_encoding}")
                    df = pd.read_csv(file_path, encoding=detected_encoding)
                    used_encoding = detected_encoding
                except (UnicodeDecodeError, LookupError):
                    self.logger.warning(f"Detected encoding '{detected_encoding}' also failed")
                    detected_encoding = None
            
            # Step 2: If automatic detection failed or didn't work, try multiple encodings
            if 'df' not in locals() or detected_encoding is None:
                df, used_encoding = self.try_multiple_encodings(file_path)
                if df is None:
                    raise ValueError("Could not read file with any known encoding. Please check file encoding.")
        
        # Store data and encoding info in state
        self.state.raw_df = df
        self.state.file_encoding = used_encoding  # Store for reference
        
        message = f"Successfully loaded DataFrame with shape: {df.shape} using {used_encoding} encoding"
        self.logger.info(message)
        return message



