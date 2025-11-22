"""
Anomaly detection tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List
from celery import current_task
from ..celery_app import celery_app
from ..services.anomaly_service import AnomalyService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.anomaly_tasks.detect_metric_anomalies")
def detect_metric_anomalies(self, company: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Detect anomalies in financial metrics asynchronously
    
    Args:
        company: Company ticker symbol
        metrics: List of metrics to analyze
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting anomaly detection"})
        
        anomaly_service = AnomalyService()
        
        # Detect anomalies for each metric
        anomalies = []
        for i, metric in enumerate(metrics):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Analyzing metric {metric} ({i+1}/{len(metrics)})",
                    "progress": (i / len(metrics)) * 100
                }
            )
            
            metric_anomalies = anomaly_service.detect_metric_anomalies(company, metric)
            anomalies.extend(metric_anomalies)
        
        # Assess risk for detected anomalies
        self.update_state(state="PROCESSING", meta={"status": "Assessing anomaly risks"})
        risk_assessments = []
        for anomaly in anomalies:
            risk = anomaly_service.assess_anomaly_risk(anomaly)
            risk_assessments.append(risk)
        
        return {
            "company": company,
            "metrics": metrics,
            "anomalies": [anomaly.dict() for anomaly in anomalies],
            "risk_assessments": risk_assessments,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Anomaly detection failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.anomaly_tasks.pattern_anomaly_detection")
def pattern_anomaly_detection(self, company: str, data_window: int = 252) -> Dict[str, Any]:
    """
    Detect pattern-based anomalies using machine learning
    
    Args:
        company: Company ticker symbol
        data_window: Number of days to analyze
        
    Returns:
        Dictionary with pattern anomaly results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting pattern anomaly detection"})
        
        anomaly_service = AnomalyService()
        
        # Analyze pattern anomalies
        self.update_state(state="PROCESSING", meta={"status": "Analyzing patterns"})
        pattern_anomalies = anomaly_service.detect_pattern_anomalies(company, data_window)
        
        # Generate explanations
        self.update_state(state="PROCESSING", meta={"status": "Generating explanations"})
        explanations = []
        for anomaly in pattern_anomalies:
            explanation = anomaly_service.explain_anomaly(anomaly)
            explanations.append(explanation)
        
        return {
            "company": company,
            "data_window": data_window,
            "pattern_anomalies": [anomaly.dict() for anomaly in pattern_anomalies],
            "explanations": explanations,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Pattern anomaly detection failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Pattern anomaly detection failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.anomaly_tasks.systemic_risk_analysis")
def systemic_risk_analysis(self, companies: List[str]) -> Dict[str, Any]:
    """
    Analyze systemic risk across multiple companies
    
    Args:
        companies: List of company ticker symbols
        
    Returns:
        Dictionary with systemic risk analysis results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting systemic risk analysis"})
        
        anomaly_service = AnomalyService()
        
        # Analyze correlations between company anomalies
        self.update_state(state="PROCESSING", meta={"status": "Analyzing anomaly correlations"})
        correlations = anomaly_service.analyze_anomaly_correlations(companies)
        
        # Assess systemic risk
        self.update_state(state="PROCESSING", meta={"status": "Assessing systemic risk"})
        systemic_risk = anomaly_service.assess_systemic_risk(companies)
        
        return {
            "companies": companies,
            "correlations": correlations,
            "systemic_risk": systemic_risk,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Systemic risk analysis failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Systemic risk analysis failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.anomaly_tasks.update_anomaly_baselines")
def update_anomaly_baselines(self, company: str) -> Dict[str, Any]:
    """
    Update baseline patterns for anomaly detection
    
    Args:
        company: Company ticker symbol
        
    Returns:
        Dictionary with baseline update results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Updating anomaly baselines"})
        
        anomaly_service = AnomalyService()
        
        # Update baselines for company
        updated_baselines = anomaly_service.update_company_baselines(company)
        
        return {
            "company": company,
            "updated_baselines": updated_baselines,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Baseline update failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Baseline update failed"}
        )
        raise