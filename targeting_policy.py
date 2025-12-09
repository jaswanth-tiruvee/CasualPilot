"""
Optimal Targeting Policy Calculator
Determines the optimal targeting strategy based on HTE predictions
"""

import numpy as np
import pandas as pd

class TargetingPolicy:
    """Calculates optimal targeting policies based on HTE."""
    
    def __init__(self, data, hte_predictions, treatment_cost=1.0):
        """
        Args:
            data: DataFrame with customer data
            hte_predictions: Array of predicted HTE for each customer
            treatment_cost: Cost per treatment (default: 1.0)
        """
        self.data = data.copy()
        self.hte_predictions = hte_predictions
        self.treatment_cost = treatment_cost
        
        # Add HTE predictions to data
        if 'predicted_hte' not in self.data.columns:
            self.data['predicted_hte'] = self.hte_predictions
    
    def calculate_roi(self, target_threshold=None, target_percentile=None):
        """
        Calculate ROI for different targeting strategies.
        
        Args:
            target_threshold: Minimum HTE threshold for targeting
            target_percentile: Top percentile to target (0-100)
        
        Returns:
            Dictionary with ROI metrics
        """
        if target_threshold is None and target_percentile is None:
            # Calculate for various thresholds
            thresholds = np.percentile(self.hte_predictions, 
                                     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            results = []
            
            for threshold in thresholds:
                mask = self.hte_predictions >= threshold
                n_targeted = np.sum(mask)
                
                if n_targeted == 0:
                    continue
                
                total_effect = np.sum(self.hte_predictions[mask])
                total_cost = n_targeted * self.treatment_cost
                roi = (total_effect - total_cost) / total_cost if total_cost > 0 else 0
                
                results.append({
                    'threshold': threshold,
                    'n_targeted': n_targeted,
                    'percentage_targeted': n_targeted / len(self.hte_predictions) * 100,
                    'total_effect': total_effect,
                    'total_cost': total_cost,
                    'net_benefit': total_effect - total_cost,
                    'roi': roi
                })
            
            return pd.DataFrame(results)
        
        elif target_threshold is not None:
            mask = self.hte_predictions >= target_threshold
        elif target_percentile is not None:
            threshold = np.percentile(self.hte_predictions, 100 - target_percentile)
            mask = self.hte_predictions >= threshold
        else:
            raise ValueError("Must specify either threshold or percentile")
        
        n_targeted = np.sum(mask)
        total_effect = np.sum(self.hte_predictions[mask])
        total_cost = n_targeted * self.treatment_cost
        roi = (total_effect - total_cost) / total_cost if total_cost > 0 else 0
        
        return {
            'n_targeted': n_targeted,
            'percentage_targeted': n_targeted / len(self.hte_predictions) * 100,
            'total_effect': total_effect,
            'total_cost': total_cost,
            'net_benefit': total_effect - total_cost,
            'roi': roi
        }
    
    def find_optimal_threshold(self):
        """Find the optimal targeting threshold that maximizes ROI."""
        thresholds = np.sort(np.unique(self.hte_predictions))
        best_roi = -np.inf
        best_threshold = None
        best_metrics = None
        
        for threshold in thresholds:
            metrics = self.calculate_roi(target_threshold=threshold)
            if metrics['roi'] > best_roi:
                best_roi = metrics['roi']
                best_threshold = threshold
                best_metrics = metrics
        
        return {
            'optimal_threshold': best_threshold,
            'optimal_roi': best_roi,
            'metrics': best_metrics
        }
    
    def compare_with_random_targeting(self, target_percentage=50):
        """
        Compare optimal targeting with random targeting.
        
        Args:
            target_percentage: Percentage of customers to target
        
        Returns:
            Dictionary with comparison metrics
        """
        # Optimal targeting (top HTE)
        n_target = int(len(self.hte_predictions) * target_percentage / 100)
        sorted_idx = np.argsort(-self.hte_predictions)
        optimal_hte = self.hte_predictions[sorted_idx[:n_target]]
        optimal_effect = np.sum(optimal_hte)
        optimal_cost = n_target * self.treatment_cost
        optimal_roi = (optimal_effect - optimal_cost) / optimal_cost if optimal_cost > 0 else 0
        
        # Random targeting
        np.random.seed(42)
        random_idx = np.random.choice(len(self.hte_predictions), n_target, replace=False)
        random_hte = self.hte_predictions[random_idx]
        random_effect = np.sum(random_hte)
        random_cost = n_target * self.treatment_cost
        random_roi = (random_effect - random_cost) / random_cost if random_cost > 0 else 0
        
        # Improvement
        improvement = ((optimal_roi - random_roi) / abs(random_roi) * 100) if random_roi != 0 else 0
        
        return {
            'target_percentage': target_percentage,
            'optimal': {
                'effect': optimal_effect,
                'cost': optimal_cost,
                'roi': optimal_roi,
                'net_benefit': optimal_effect - optimal_cost
            },
            'random': {
                'effect': random_effect,
                'cost': random_cost,
                'roi': random_roi,
                'net_benefit': random_effect - random_cost
            },
            'improvement_percentage': improvement,
            'absolute_improvement': optimal_effect - random_effect
        }
    
    def get_targeting_list(self, threshold=None, percentile=None, top_n=None):
        """
        Get list of customers to target.
        
        Args:
            threshold: Minimum HTE threshold
            percentile: Top percentile to target
            top_n: Top N customers to target
        
        Returns:
            DataFrame with targeted customers
        """
        if threshold is not None:
            mask = self.hte_predictions >= threshold
        elif percentile is not None:
            threshold = np.percentile(self.hte_predictions, 100 - percentile)
            mask = self.hte_predictions >= threshold
        elif top_n is not None:
            sorted_idx = np.argsort(-self.hte_predictions)
            mask = np.zeros(len(self.hte_predictions), dtype=bool)
            mask[sorted_idx[:top_n]] = True
        else:
            raise ValueError("Must specify threshold, percentile, or top_n")
        
        targeted = self.data[mask].copy()
        targeted = targeted.sort_values('predicted_hte', ascending=False)
        
        return targeted

