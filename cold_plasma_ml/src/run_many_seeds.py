#!/usr/bin/env python
"""
Run the ML pipeline with multiple random seeds and aggregate results.
"""
import argparse
import pandas as pd
import mlflow
from tqdm import tqdm
from main import main as run_pipeline
from config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI

def run_multiple_seeds(seeds, experiment_name=None):
    """
    Run the pipeline with multiple random seeds and aggregate results.
    
    Args:
        seeds: List of random seeds to use
        experiment_name: Optional name for the experiment
    
    Returns:
        DataFrame with aggregated results across all seeds
    """
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    all_results = []
    
    for seed in tqdm(seeds, desc="Running with different seeds"):
        try:
            # Run the pipeline with the current seed
            results = run_pipeline(random_state=seed)
            
            # Add seed to results
            results['seed'] = seed
            all_results.append(results)
            
        except Exception as e:
            print(f"Error with seed {seed}: {str(e)}")
            continue
    
    # Combine all results
    if not all_results:
        raise ValueError("No successful runs completed.")
        
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate mean and std across seeds for each model
    metrics = ['test_MAE', 'test_RMSE', 'test_R2']
    stats = []
    
    for model in combined_results['model'].unique():
        model_results = combined_results[combined_results['model'] == model]
        
        # Calculate statistics
        stats.append({
            'model': model,
            'runs': len(model_results),
            **{f'{metric}_mean': model_results[metric].mean() for metric in metrics},
            **{f'{metric}_std': model_results[metric].std() for metric in metrics},
            'seeds': ','.join(map(str, model_results['seed'].tolist()))
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Log the aggregated results
    with mlflow.start_run(run_name="AGGREGATED_RESULTS"):
        # Log parameters
        mlflow.log_param("num_seeds", len(seeds))
        mlflow.log_param("seeds", ",".join(map(str, seeds)))
        
        # Log metrics (mean and std for each model)
        for _, row in stats_df.iterrows():
            model = row['model']
            for metric in metrics:
                mlflow.log_metric(f"{model}_{metric}_mean", row[f"{metric}_mean"])
                mlflow.log_metric(f"{model}_{metric}_std", row[f"{metric}_std"])
        
        # Save detailed results
        mlflow.log_text(combined_results.to_csv(index=False), "all_results.csv")
        mlflow.log_text(stats_df.to_csv(index=False), "aggregated_results.csv")
        
        # Create and log plots
        import matplotlib.pyplot as plt
        
        # Plot performance across seeds
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for model in combined_results['model'].unique():
                model_data = combined_results[combined_results['model'] == model]
                plt.plot(model_data['seed'], model_data[metric], 'o-', label=model)
            
            plt.title(f"{metric} across different random seeds")
            plt.xlabel("Random Seed")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Log the plot
            mlflow.log_figure(plt.gcf(), f"{metric}_across_seeds.png")
            plt.close()
    
    return combined_results, stats_df

def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline with multiple random seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112],
                       help="List of random seeds to use")
    parser.add_argument("--experiment", type=str, default=f"{EXPERIMENT_NAME}_robust",
                       help="Name for the MLflow experiment")
    
    args = parser.parse_args()
    
    print(f"Running pipeline with seeds: {args.seeds}")
    print(f"MLflow experiment: {args.experiment}")
    
    combined_results, stats_df = run_multiple_seeds(args.seeds, args.experiment)
    
    print("\nAggregated Results:")
    print("=" * 50)
    print(stats_df.to_string(index=False))
    
    print("\nDetailed results and plots are available in MLflow UI.")
    print(f"To view: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()
