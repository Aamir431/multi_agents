# main.py
import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Run Data Science Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--target', type=str, required=True, help='Name of the target variable column')  # ADDED THIS LINE
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate data path exists
    if not os.path.exists(args.data_path):
        print(f" Data file not found: {args.data_path}")
        exit(1)
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f" Config file not found: {args.config}")
        print("Creating default config...")
        # You can add code here to create default config if needed
        exit(1)
    
    try:
        # Import locally to avoid circular imports
        from orchestrator import run_simple_pipeline
        from result_manager import ResultsManager
        
        # Initialize
        results_manager = ResultsManager(output_dir=args.output_dir)
        
        # Run pipeline
        print(" Starting Data Science Pipeline...")
        print(f" Data: {args.data_path}")
        print(f" Target: {args.target}")  # ADDED THIS LINE
        print(f" Config: {args.config}")
        print(f" Output: {args.output_dir}")
        print("-" * 50)
        
        # Run pipeline with target variable - MODIFIED THIS LINE
        results = run_simple_pipeline(args.data_path, args.config, target_variable=args.target)
        
        # Save results
        results_dir = results_manager.save_results(results)
        
        print(f"\n Pipeline completed successfully!")
        print(f" Results saved to: {results_dir}")
        
        # Show summary of results
        if results and 'model_training_result' in results:
            model_result = results['model_training_result']
            if hasattr(model_result, 'metrics') and model_result.metrics:
                print("\n📈 Model Performance Metrics:")
                for metric, value in model_result.metrics.items():
                    print(f"   {metric}: {value:.3f}")
        
        return 0  # Success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all required files exist:")
        print("- results_manager.py")
        print("- logger.py (optional)")
        print("- orchestrator.py")
        print("- config.yaml")
        print("- agents/ directory with all agent files")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print("Use --verbose for detailed error information")
        return 1

if __name__ == "__main__":
    exit(main())