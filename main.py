"""
Main Script: CausalPilot - The Treatment Effect Simulator
Run this script to execute the full causal inference pipeline
"""

import numpy as np
import pandas as pd
from data_generator import MarketingDataGenerator
from causal_pipeline import CausalInferencePipeline
from visualization import HTEVisualizer
from targeting_policy import TargetingPolicy

def main():
    print("=" * 70)
    print("CausalPilot: The Treatment Effect Simulator")
    print("=" * 70)
    
    # Step 1: Generate Data
    print("\n[1/5] Generating synthetic marketing data...")
    generator = MarketingDataGenerator(n_samples=5000, seed=42)
    data = generator.generate()
    print(f"✓ Generated {len(data)} samples")
    print(f"  - Treatment rate: {data['treatment'].mean()*100:.1f}%")
    print(f"  - Conversion rate: {data['outcome'].mean()*100:.1f}%")
    
    # Step 2: Identify Causal Effect
    print("\n[2/5] Identifying causal effect with DoWhy...")
    pipeline = CausalInferencePipeline(data, treatment='treatment', outcome='outcome')
    confounders = ['age', 'income', 'purchase_history', 'engagement_score']
    identified_estimand = pipeline.identify_causal_effect(confounders=confounders)
    print("✓ Causal effect identified")
    
    # Step 3: Estimate HTE
    print("\n[3/5] Estimating Heterogeneous Treatment Effects...")
    method = 'dml'  # Can be 'dml', 'forest', 'tlearner', 'slearner', 'xlearner'
    hte_predictions, X_test = pipeline.estimate_hte(method=method)
    print(f"✓ HTE estimated using {method.upper()}")
    print(f"  - Average Treatment Effect: {pipeline.ate:.4f}")
    print(f"  - HTE range: [{np.min(hte_predictions):.4f}, {np.max(hte_predictions):.4f}]")
    
    # Step 4: Analyze Targeting Segments
    print("\n[4/5] Analyzing targeting segments...")
    segments = pipeline.get_targeting_segments(n_segments=4)
    print("✓ Segmentation complete")
    for seg in segments:
        print(f"  Segment {seg['segment']}: HTE={seg['hte_mean']:.4f}, "
              f"Size={seg['size']} ({seg['percentage']:.1f}%)")
    
    # Step 5: Calculate Optimal Targeting Policy
    print("\n[5/5] Calculating optimal targeting policy...")
    treatment_cost = 1.0
    policy = TargetingPolicy(data, pipeline.hte_predictions_full, 
                            treatment_cost=treatment_cost)
    optimal = policy.find_optimal_threshold()
    comparison = policy.compare_with_random_targeting(target_percentage=50)
    
    print("✓ Optimal targeting policy calculated")
    print(f"\n--- Optimal Targeting Results ---")
    print(f"Optimal HTE Threshold: {optimal['optimal_threshold']:.4f}")
    print(f"Optimal ROI: {optimal['optimal_roi']*100:.2f}%")
    print(f"Targeting {optimal['metrics']['percentage_targeted']:.1f}% of customers")
    
    print(f"\n--- Comparison: Optimal vs Random (50% targeting) ---")
    print(f"Optimal ROI: {comparison['optimal']['roi']*100:.2f}%")
    print(f"Random ROI: {comparison['random']['roi']*100:.2f}%")
    print(f"Improvement: {comparison['improvement_percentage']:.1f}%")
    print(f"Additional Effect: +{comparison['absolute_improvement']:.4f}")
    
    # Calculate simulated ROI improvement
    baseline_roi = comparison['random']['roi']
    optimal_roi = comparison['optimal']['roi']
    roi_improvement = ((optimal_roi - baseline_roi) / abs(baseline_roi)) * 100
    print(f"\n🎯 Simulated ROI Boost: {roi_improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)
    
    # Save results
    results_df = data.copy()
    results_df['predicted_hte'] = pipeline.hte_predictions_full
    results_df.to_csv('causalpilot_results.csv', index=False)
    print("\n✓ Results saved to 'causalpilot_results.csv'")
    
    return data, pipeline, policy

if __name__ == "__main__":
    data, pipeline, policy = main()

