"""
Synthetic Marketing Dataset Generator
Creates realistic marketing data with treatment, outcome, and confounders
for Heterogeneous Treatment Effect (HTE) modeling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MarketingDataGenerator:
    """Generates synthetic marketing data with known causal structure."""
    
    def __init__(self, n_samples=5000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
    
    def generate(self):
        """Generate synthetic marketing dataset."""
        
        # Confounders (affect both treatment and outcome)
        age = np.random.normal(35, 12, self.n_samples).clip(18, 80)
        income = np.random.lognormal(10.5, 0.5, self.n_samples)
        purchase_history = np.random.poisson(3, self.n_samples)
        engagement_score = np.random.beta(2, 5, self.n_samples) * 100
        
        # Additional features (only affect outcome, not treatment assignment)
        product_category_pref = np.random.choice(['A', 'B', 'C'], self.n_samples)
        website_visits = np.random.poisson(10, self.n_samples)
        time_on_site = np.random.gamma(5, 2, self.n_samples)
        
        # Treatment assignment (affected by confounders)
        # Higher income and engagement = higher treatment probability
        treatment_prob = (
            0.2 + 
            0.3 * (income > np.percentile(income, 60)) + 
            0.2 * (engagement_score > 40) +
            0.1 * (age > 30) * (age < 50)
        )
        treatment_prob = np.clip(treatment_prob, 0, 1)
        treatment = np.random.binomial(1, treatment_prob, self.n_samples)
        
        # Outcome generation with heterogeneous treatment effects
        # Base conversion probability
        base_prob = (
            0.05 +
            0.02 * (age > 25) * (age < 45) +
            0.03 * (income > np.percentile(income, 50)) +
            0.02 * (engagement_score > 30) +
            0.01 * purchase_history / 5
        )
        
        # Heterogeneous Treatment Effect (HTE)
        # Treatment effect varies by customer characteristics
        hte = (
            0.08 +  # Base treatment effect
            0.05 * (engagement_score > 50) +  # High engagement responds better
            0.03 * (income > np.percentile(income, 70)) +  # High income responds better
            0.04 * (age > 30) * (age < 50) -  # Middle age responds better
            0.02 * (purchase_history > 5)  # Too many purchases = lower response
        )
        hte = np.clip(hte, 0.01, 0.25)
        
        # Outcome probability
        outcome_prob = base_prob + treatment * hte
        outcome_prob = np.clip(outcome_prob, 0, 1)
        outcome = np.random.binomial(1, outcome_prob, self.n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'purchase_history': purchase_history,
            'engagement_score': engagement_score,
            'product_category_pref': product_category_pref,
            'website_visits': website_visits,
            'time_on_site': time_on_site,
            'treatment': treatment,
            'outcome': outcome,
            'true_hte': hte  # For evaluation purposes
        })
        
        return df
    
    def get_feature_names(self):
        """Return list of feature names (confounders and covariates)."""
        return ['age', 'income', 'purchase_history', 'engagement_score', 
                'product_category_pref', 'website_visits', 'time_on_site']

