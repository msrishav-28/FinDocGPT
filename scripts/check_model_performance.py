#!/usr/bin/env python3
"""
Script to check model performance and determine if retraining is needed.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests


class ModelPerformanceChecker:
    """Checks model performance and determines retraining needs."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_model_metrics(self, model_type: str, days: int = 7) -> Dict[str, Any]:
        """Get model performance metrics from the API."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'model_type': model_type,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        
        response = requests.get(
            f'{self.api_url}/api/admin/monitoring/model-performance',
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def check_sentiment_model(self) -> Dict[str, Any]:
        """Check sentiment analysis model performance."""
        metrics = self.get_model_metrics('sentiment')
        
        # Performance thresholds
        min_accuracy = 0.85
        max_response_time = 500  # ms
        min_predictions_per_day = 100
        
        accuracy = metrics.get('accuracy', 0)
        avg_response_time = metrics.get('avg_response_time', float('inf'))
        predictions_per_day = metrics.get('predictions_per_day', 0)
        
        needs_retraining = (
            accuracy < min_accuracy or
            avg_response_time > max_response_time or
            predictions_per_day < min_predictions_per_day
        )
        
        return {
            'model_type': 'sentiment',
            'needs_retraining': needs_retraining,
            'metrics': {
                'accuracy': accuracy,
                'avg_response_time': avg_response_time,
                'predictions_per_day': predictions_per_day
            },
            'thresholds': {
                'min_accuracy': min_accuracy,
                'max_response_time': max_response_time,
                'min_predictions_per_day': min_predictions_per_day
            }
        }
    
    def check_forecast_model(self) -> Dict[str, Any]:
        """Check forecasting model performance."""
        metrics = self.get_model_metrics('forecast')
        
        # Performance thresholds
        max_mape = 0.15  # Mean Absolute Percentage Error
        max_response_time = 2000  # ms
        min_predictions_per_day = 50
        
        mape = metrics.get('mape', float('inf'))
        avg_response_time = metrics.get('avg_response_time', float('inf'))
        predictions_per_day = metrics.get('predictions_per_day', 0)
        
        needs_retraining = (
            mape > max_mape or
            avg_response_time > max_response_time or
            predictions_per_day < min_predictions_per_day
        )
        
        return {
            'model_type': 'forecast',
            'needs_retraining': needs_retraining,
            'metrics': {
                'mape': mape,
                'avg_response_time': avg_response_time,
                'predictions_per_day': predictions_per_day
            },
            'thresholds': {
                'max_mape': max_mape,
                'max_response_time': max_response_time,
                'min_predictions_per_day': min_predictions_per_day
            }
        }
    
    def check_anomaly_model(self) -> Dict[str, Any]:
        """Check anomaly detection model performance."""
        metrics = self.get_model_metrics('anomaly')
        
        # Performance thresholds
        min_precision = 0.80
        min_recall = 0.75
        max_response_time = 1000  # ms
        
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        avg_response_time = metrics.get('avg_response_time', float('inf'))
        
        needs_retraining = (
            precision < min_precision or
            recall < min_recall or
            avg_response_time > max_response_time
        )
        
        return {
            'model_type': 'anomaly',
            'needs_retraining': needs_retraining,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'avg_response_time': avg_response_time
            },
            'thresholds': {
                'min_precision': min_precision,
                'min_recall': min_recall,
                'max_response_time': max_response_time
            }
        }
    
    def check_recommendation_model(self) -> Dict[str, Any]:
        """Check recommendation model performance."""
        metrics = self.get_model_metrics('recommendation')
        
        # Performance thresholds
        min_click_through_rate = 0.10
        min_conversion_rate = 0.05
        max_response_time = 800  # ms
        
        ctr = metrics.get('click_through_rate', 0)
        conversion_rate = metrics.get('conversion_rate', 0)
        avg_response_time = metrics.get('avg_response_time', float('inf'))
        
        needs_retraining = (
            ctr < min_click_through_rate or
            conversion_rate < min_conversion_rate or
            avg_response_time > max_response_time
        )
        
        return {
            'model_type': 'recommendation',
            'needs_retraining': needs_retraining,
            'metrics': {
                'click_through_rate': ctr,
                'conversion_rate': conversion_rate,
                'avg_response_time': avg_response_time
            },
            'thresholds': {
                'min_click_through_rate': min_click_through_rate,
                'min_conversion_rate': min_conversion_rate,
                'max_response_time': max_response_time
            }
        }
    
    def check_all_models(self, model_types: List[str]) -> Dict[str, Any]:
        """Check performance for all specified models."""
        model_checkers = {
            'sentiment': self.check_sentiment_model,
            'forecast': self.check_forecast_model,
            'anomaly': self.check_anomaly_model,
            'recommendation': self.check_recommendation_model
        }
        
        results = {}
        models_needing_retraining = []
        
        for model_type in model_types:
            if model_type in model_checkers:
                try:
                    result = model_checkers[model_type]()
                    results[model_type] = result
                    
                    if result['needs_retraining']:
                        models_needing_retraining.append(model_type)
                        
                except Exception as e:
                    print(f"Error checking {model_type} model: {e}", file=sys.stderr)
                    results[model_type] = {
                        'model_type': model_type,
                        'needs_retraining': False,
                        'error': str(e)
                    }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'models_checked': model_types,
            'models_needing_retraining': models_needing_retraining,
            'needs_retraining': len(models_needing_retraining) > 0,
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(description='Check model performance for retraining')
    parser.add_argument('--api-url', required=True, help='API base URL')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    parser.add_argument('--model-type', default='all', 
                       choices=['all', 'sentiment', 'forecast', 'anomaly', 'recommendation'],
                       help='Type of model to check')
    parser.add_argument('--output-format', default='json', choices=['json', 'github'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Determine which models to check
    if args.model_type == 'all':
        model_types = ['sentiment', 'forecast', 'anomaly', 'recommendation']
    else:
        model_types = [args.model_type]
    
    # Check model performance
    checker = ModelPerformanceChecker(args.api_url, args.api_key)
    results = checker.check_all_models(model_types)
    
    # Output results
    if args.output_format == 'github':
        # Output for GitHub Actions
        print(f"::set-output name=needs_retraining::{str(results['needs_retraining']).lower()}")
        print(f"::set-output name=models_to_retrain::{json.dumps(results['models_needing_retraining'])}")
        
        # Create summary
        if results['needs_retraining']:
            print(f"Models needing retraining: {', '.join(results['models_needing_retraining'])}")
        else:
            print("No models need retraining at this time.")
    else:
        # JSON output
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if not results['needs_retraining'] else 1)


if __name__ == '__main__':
    main()