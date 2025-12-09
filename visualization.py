"""
Visualization Module for HTE Analysis
Creates interactive plots using Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class HTEVisualizer:
    """Creates visualizations for Heterogeneous Treatment Effect analysis."""
    
    def __init__(self, data, hte_predictions):
        self.data = data.copy()
        self.hte_predictions = hte_predictions
        
        # Add HTE predictions to data
        if 'predicted_hte' not in self.data.columns:
            self.data['predicted_hte'] = self.hte_predictions
    
    def plot_hte_distribution(self, true_hte=None):
        """Plot the distribution of Heterogeneous Treatment Effects."""
        fig = go.Figure()
        
        # Histogram of predicted HTE
        fig.add_trace(go.Histogram(
            x=self.hte_predictions,
            nbinsx=50,
            name='Predicted HTE',
            opacity=0.7,
            marker_color='blue'
        ))
        
        # Overlay true HTE if available
        if true_hte is not None:
            fig.add_trace(go.Histogram(
                x=true_hte,
                nbinsx=50,
                name='True HTE',
                opacity=0.5,
                marker_color='red'
            ))
        
        fig.update_layout(
            title='Distribution of Heterogeneous Treatment Effects',
            xaxis_title='Treatment Effect',
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_hte_by_feature(self, feature_name, n_bins=10):
        """Plot HTE by feature values."""
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature {feature_name} not found in data")
        
        # Create bins
        feature_values = self.data[feature_name]
        if feature_values.dtype in ['object', 'category']:
            # Categorical feature
            grouped = self.data.groupby(feature_name)['predicted_hte'].agg(['mean', 'std', 'count'])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=grouped.index,
                y=grouped['mean'],
                error_y=dict(type='data', array=grouped['std']),
                name='Average HTE',
                marker_color='steelblue'
            ))
            fig.update_layout(
                title=f'Average HTE by {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Average Treatment Effect',
                template='plotly_white',
                height=500
            )
        else:
            # Numerical feature
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=self.hte_predictions,
                mode='markers',
                marker=dict(
                    size=5,
                    opacity=0.6,
                    color=self.hte_predictions,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="HTE")
                ),
                name='HTE'
            ))
            
            # Add trend line
            z = np.polyfit(feature_values, self.hte_predictions, 1)
            p = np.poly1d(z)
            sorted_idx = np.argsort(feature_values)
            fig.add_trace(go.Scatter(
                x=feature_values.iloc[sorted_idx],
                y=p(feature_values.iloc[sorted_idx]),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'HTE Distribution by {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Treatment Effect',
                template='plotly_white',
                height=500
            )
        
        return fig
    
    def plot_targeting_segments(self, segments):
        """Visualize targeting segments."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('HTE by Segment', 'Segment Sizes'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart of average HTE per segment
        segment_nums = [s['segment'] for s in segments]
        hte_means = [s['hte_mean'] for s in segments]
        
        fig.add_trace(
            go.Bar(
                x=segment_nums,
                y=hte_means,
                name='Avg HTE',
                marker_color='steelblue',
                text=[f'{h:.4f}' for h in hte_means],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Pie chart of segment sizes
        segment_labels = [f"Segment {s['segment']}" for s in segments]
        segment_sizes = [s['percentage'] for s in segments]
        
        fig.add_trace(
            go.Pie(
                labels=segment_labels,
                values=segment_sizes,
                name="Segments",
                textinfo='label+percent'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Targeting Segmentation Analysis',
            template='plotly_white',
            height=500
        )
        
        fig.update_xaxes(title_text="Segment", row=1, col=1)
        fig.update_yaxes(title_text="Average HTE", row=1, col=1)
        
        return fig
    
    def plot_uplift_curve(self, top_percentile_range=np.arange(0, 101, 5)):
        """Plot uplift curve showing cumulative effect of targeting."""
        sorted_idx = np.argsort(-self.hte_predictions)  # Sort descending
        sorted_hte = self.hte_predictions[sorted_idx]
        
        cumulative_effects = []
        percentiles = []
        
        for p in top_percentile_range:
            n_top = int(len(sorted_hte) * p / 100)
            if n_top > 0:
                cumulative_effect = np.sum(sorted_hte[:n_top])
                cumulative_effects.append(cumulative_effect)
                percentiles.append(p)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=cumulative_effects,
            mode='lines+markers',
            name='Cumulative HTE',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        fig.update_layout(
            title='Uplift Curve: Cumulative Treatment Effect by Targeting Percentile',
            xaxis_title='Top Percentile Targeted (%)',
            yaxis_title='Cumulative Treatment Effect',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores):
        """Plot feature importance for HTE prediction."""
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_scores,
            y=sorted_features,
            orientation='h',
            marker_color='steelblue',
            text=[f'{s:.3f}' for s in sorted_scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Feature Importance for HTE Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            height=500
        )
        
        return fig

