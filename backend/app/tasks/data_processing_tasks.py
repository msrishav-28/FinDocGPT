"""
Data processing tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List, Optional
from celery import current_task
from ..celery_app import celery_app
from ..services.data_integration_service import DataIntegrationService
from ..services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.update_market_data")
def update_market_data(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Update market data for specified tickers or all tracked tickers
    
    Args:
        tickers: Optional list of ticker symbols to update
        
    Returns:
        Dictionary with market data update results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting market data update"})
        
        market_service = MarketDataService()
        
        # Get list of tickers to update
        if not tickers:
            tickers = market_service.get_tracked_tickers()
        
        updated_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Updating data for {ticker} ({i+1}/{len(tickers)})",
                        "progress": (i / len(tickers)) * 100
                    }
                )
                
                # Update market data for ticker
                update_result = market_service.update_ticker_data(ticker)
                updated_tickers.append({
                    "ticker": ticker,
                    "result": update_result
                })
                
            except Exception as e:
                logger.error(f"Failed to update data for {ticker}: {str(e)}")
                failed_tickers.append({
                    "ticker": ticker,
                    "error": str(e)
                })
        
        return {
            "total_tickers": len(tickers),
            "successful_updates": len(updated_tickers),
            "failed_updates": len(failed_tickers),
            "updated_tickers": updated_tickers,
            "failed_tickers": failed_tickers,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Market data update failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Market data update failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.sync_external_data")
def sync_external_data(self, data_sources: List[str]) -> Dict[str, Any]:
    """
    Synchronize data from external sources
    
    Args:
        data_sources: List of data source names to sync
        
    Returns:
        Dictionary with synchronization results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting external data sync"})
        
        data_service = DataIntegrationService()
        
        sync_results = {}
        
        for i, source in enumerate(data_sources):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Syncing {source} ({i+1}/{len(data_sources)})",
                        "progress": (i / len(data_sources)) * 100
                    }
                )
                
                # Sync data from source
                result = data_service.sync_data_source(source)
                sync_results[source] = {
                    "status": "success",
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Failed to sync {source}: {str(e)}")
                sync_results[source] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "data_sources": data_sources,
            "sync_results": sync_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"External data sync failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "External data sync failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.process_financial_reports")
def process_financial_reports(self, company: str, report_type: str) -> Dict[str, Any]:
    """
    Process financial reports for a company
    
    Args:
        company: Company ticker symbol
        report_type: Type of report (earnings, 10k, 10q, etc.)
        
    Returns:
        Dictionary with processing results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting financial report processing"})
        
        data_service = DataIntegrationService()
        
        # Fetch latest reports
        self.update_state(state="PROCESSING", meta={"status": "Fetching latest reports"})
        reports = data_service.fetch_financial_reports(company, report_type)
        
        # Process each report
        processed_reports = []
        for i, report in enumerate(reports):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Processing report {i+1}/{len(reports)}",
                    "progress": (i / len(reports)) * 100
                }
            )
            
            processed_report = data_service.process_financial_report(report)
            processed_reports.append(processed_report)
        
        return {
            "company": company,
            "report_type": report_type,
            "total_reports": len(reports),
            "processed_reports": processed_reports,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Financial report processing failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Report processing failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.calculate_financial_metrics")
def calculate_financial_metrics(self, company: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Calculate financial metrics for a company
    
    Args:
        company: Company ticker symbol
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with calculated metrics
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting financial metrics calculation"})
        
        data_service = DataIntegrationService()
        
        # Calculate metrics
        calculated_metrics = {}
        for i, metric in enumerate(metrics):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Calculating {metric} ({i+1}/{len(metrics)})",
                    "progress": (i / len(metrics)) * 100
                }
            )
            
            try:
                metric_value = data_service.calculate_metric(company, metric)
                calculated_metrics[metric] = metric_value
            except Exception as e:
                logger.error(f"Failed to calculate {metric}: {str(e)}")
                calculated_metrics[metric] = {"error": str(e)}
        
        return {
            "company": company,
            "metrics": metrics,
            "calculated_metrics": calculated_metrics,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Financial metrics calculation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Metrics calculation failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.data_quality_check")
def data_quality_check(self, data_type: str, company: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform data quality checks
    
    Args:
        data_type: Type of data to check (market_data, financial_reports, etc.)
        company: Optional company to check data for
        
    Returns:
        Dictionary with data quality results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting data quality check"})
        
        data_service = DataIntegrationService()
        
        # Perform quality checks
        quality_results = data_service.check_data_quality(data_type, company)
        
        return {
            "data_type": data_type,
            "company": company,
            "quality_results": quality_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Data quality check failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Data quality check failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.data_processing_tasks.cleanup_old_data")
def cleanup_old_data(self, data_type: str, retention_days: int = 365) -> Dict[str, Any]:
    """
    Clean up old data based on retention policy
    
    Args:
        data_type: Type of data to clean up
        retention_days: Number of days to retain data
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting data cleanup"})
        
        data_service = DataIntegrationService()
        
        # Perform data cleanup
        cleanup_results = data_service.cleanup_old_data(data_type, retention_days)
        
        return {
            "data_type": data_type,
            "retention_days": retention_days,
            "cleanup_results": cleanup_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Data cleanup failed"}
        )
        raise