"""
Causal Inference Pipeline
Implements the full causal inference workflow using DoWhy and EconML
"""

import numpy as np
import pandas as pd
from dowhy import CausalModel
from econml.dml import CausalForestDML, LinearDML
from econml.metalearners import TLearner, SLearner, XLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class CausalInferencePipeline:
    """Main pipeline for causal inference and HTE estimation."""
    
    def __init__(self, data, treatment='treatment', outcome='outcome'):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.causal_model = None
        self.hte_estimator = None
        self.hte_predictions = None
        
    def identify_causal_effect(self, confounders=None):
        """
        Use DoWhy to identify the causal effect.
        
        Args:
            confounders: List of confounder variable names
        """
        if confounders is None:
            # Auto-detect confounders (exclude treatment and outcome)
            confounders = [col for col in self.data.columns 
                          if col not in [self.treatment, self.outcome, 'true_hte']]
        
        # Define causal graph
        graph = self._create_causal_graph(confounders)
        
        # Create DoWhy causal model
        self.causal_model = CausalModel(
            data=self.data,
            treatment=self.treatment,
            outcome=self.outcome,
            graph=graph
        )
        
        # Identify causal effect
        identified_estimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )
        
        return identified_estimand
    
    def _create_causal_graph(self, confounders):
        """Create a causal graph string for DoWhy."""
        nodes = confounders + [self.treatment, self.outcome]
        edges = []
        
        # Confounders affect treatment
        for conf in confounders:
            edges.append(f"{conf} -> {self.treatment}")
        
        # Confounders affect outcome
        for conf in confounders:
            edges.append(f"{conf} -> {self.outcome}")
        
        # Treatment affects outcome
        edges.append(f"{self.treatment} -> {self.outcome}")
        
        graph = "digraph {" + "; ".join(edges) + "}"
        return graph
    
    def estimate_hte(self, method='dml', X=None, T=None, y=None):
        """
        Estimate Heterogeneous Treatment Effects using EconML.
        
        Args:
            method: 'dml', 'forest', 'tlearner', 'slearner', or 'xlearner'
            X: Features matrix (if None, uses all non-treatment/outcome columns)
            T: Treatment vector (if None, uses self.treatment)
            y: Outcome vector (if None, uses self.outcome)
        """
        # Prepare data
        if X is None:
            X = self.data.drop([self.treatment, self.outcome, 'true_hte'], 
                              axis=1, errors='ignore')
            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)
        
        if T is None:
            T = self.data[self.treatment].values
        
        if y is None:
            y = self.data[self.outcome].values
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(T, pd.Series):
            T = T.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
            X, T, y, test_size=0.3, random_state=42
        )
        
        # Choose estimator
        if method == 'dml':
            self.hte_estimator = LinearDML(
                model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                model_t=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                discrete_treatment=True,
                cv=5
            )
        elif method == 'forest':
            self.hte_estimator = CausalForestDML(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                discrete_treatment=True,
                random_state=42
            )
        elif method == 'tlearner':
            self.hte_estimator = TLearner(
                models=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            )
        elif method == 'slearner':
            self.hte_estimator = SLearner(
                overall_model=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            )
        elif method == 'xlearner':
            self.hte_estimator = XLearner(
                models=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                propensity_model=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit the estimator
        print(f"Fitting {method.upper()} estimator...")
        self.hte_estimator.fit(y_train, T_train, X=X_train)
        
        # Predict HTE on test set
        self.hte_predictions = self.hte_estimator.effect(X_test)
        
        # Also predict on full dataset for visualization
        self.hte_predictions_full = self.hte_estimator.effect(X)
        
        # Calculate average treatment effect
        self.ate = np.mean(self.hte_predictions)
        
        print(f"Average Treatment Effect (ATE): {self.ate:.4f}")
        print(f"HTE Range: [{np.min(self.hte_predictions):.4f}, {np.max(self.hte_predictions):.4f}]")
        
        return self.hte_predictions, X_test
    
    def estimate_ate(self):
        """Estimate Average Treatment Effect using DoWhy."""
        if self.causal_model is None:
            raise ValueError("Must identify causal effect first")
        
        # Estimate ATE
        causal_estimate = self.causal_model.estimate_effect(
            identified_estimand=self.causal_model.identify_effect(
                proceed_when_unidentifiable=True
            ),
            method_name="backdoor.linear_regression"
        )
        
        return causal_estimate
    
    def simulate_counterfactuals(self, X, treatment_value=1):
        """
        Simulate counterfactual outcomes.
        
        Args:
            X: Feature matrix
            treatment_value: Treatment value to simulate (0 or 1)
        """
        if self.hte_estimator is None:
            raise ValueError("Must estimate HTE first")
        
        # Predict outcomes under treatment
        if treatment_value == 1:
            # Predict with treatment
            effect = self.hte_estimator.effect(X)
            # Base outcome (from control group)
            base_outcome = self.data[self.data[self.treatment] == 0][self.outcome].mean()
            counterfactual = base_outcome + effect
        else:
            # Predict without treatment
            effect = self.hte_estimator.effect(X)
            # Base outcome (from treated group)
            base_outcome = self.data[self.data[self.treatment] == 1][self.outcome].mean()
            counterfactual = base_outcome - effect
        
        return counterfactual
    
    def get_targeting_segments(self, n_segments=4):
        """
        Create targeting segments based on HTE predictions.
        
        Args:
            n_segments: Number of segments to create
        """
        if self.hte_predictions_full is None:
            raise ValueError("Must estimate HTE first")
        
        # Create segments based on HTE quantiles
        quantiles = np.linspace(0, 1, n_segments + 1)
        segment_breaks = np.quantile(self.hte_predictions_full, quantiles)
        
        segments = []
        for i in range(n_segments):
            mask = (self.hte_predictions_full >= segment_breaks[i]) & \
                   (self.hte_predictions_full < segment_breaks[i + 1])
            if i == n_segments - 1:  # Include upper bound for last segment
                mask = self.hte_predictions_full >= segment_breaks[i]
            
            segment_data = {
                'segment': i + 1,
                'hte_min': segment_breaks[i],
                'hte_max': segment_breaks[i + 1] if i < n_segments - 1 else np.inf,
                'hte_mean': np.mean(self.hte_predictions_full[mask]),
                'size': np.sum(mask),
                'percentage': np.sum(mask) / len(self.hte_predictions_full) * 100
            }
            segments.append(segment_data)
        
        return segments

