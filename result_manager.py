# results_manager.py (if you need it)
import os
import json
import pandas as pd
from datetime import datetime

class ResultsManager:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(self, results):
        """Save pipeline results to output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results metadata
        if results:
            metadata = {
                'timestamp': timestamp,
                'success': True,
                'agents_executed': list(results.keys()) if isinstance(results, dict) else []
            }
            
            with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return results_dir