"""
Performance monitoring dashboard for system administrators.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from app.dependencies.auth import get_current_admin_user
from app.monitoring.metrics import metrics
from app.monitoring.health import health_monitor
from app.monitoring.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin/monitoring", tags=["monitoring"])


@router.get("/health")
async def get_health_status(current_user = Depends(get_current_admin_user)):
    """Get current system health status."""
    return await health_monitor.get_system_health()


@router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Specific metric name"),
    since_minutes: int = Query(60, description="Minutes of history to retrieve"),
    current_user = Depends(get_current_admin_user)
):
    """Get system metrics."""
    since = datetime.utcnow() - timedelta(minutes=since_minutes)
    return metrics.get_metrics(metric_name, since)


@router.get("/metrics/summary")
async def get_metrics_summary(current_user = Depends(get_current_admin_user)):
    """Get metrics summary."""
    return metrics.get_summary()


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(current_user = Depends(get_current_admin_user)):
    """Get monitoring dashboard HTML."""
    
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Intelligence - System Monitoring</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #2563eb;
            }
            .metric-label {
                color: #6b7280;
                font-size: 0.9em;
            }
            .status-healthy { color: #10b981; }
            .status-warning { color: #f59e0b; }
            .status-critical { color: #ef4444; }
            .chart-container {
                position: relative;
                height: 300px;
                margin-top: 20px;
            }
            .health-check {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                border-left: 4px solid #e5e7eb;
                margin-bottom: 10px;
                background: #f9fafb;
            }
            .health-check.healthy { border-left-color: #10b981; }
            .health-check.warning { border-left-color: #f59e0b; }
            .health-check.critical { border-left-color: #ef4444; }
            .refresh-btn {
                background: #2563eb;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
            }
            .refresh-btn:hover {
                background: #1d4ed8;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #e5e7eb;
            }
            th {
                background-color: #f9fafb;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Financial Intelligence System - Monitoring Dashboard</h1>
                <p>Real-time system performance and health monitoring</p>
                <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
                <span id="last-updated" style="margin-left: 20px; color: #6b7280;"></span>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>System Health</h3>
                    <div id="health-status"></div>
                </div>

                <div class="card">
                    <h3>System Resources</h3>
                    <div id="system-metrics"></div>
                </div>

                <div class="card">
                    <h3>API Performance</h3>
                    <div id="api-metrics"></div>
                </div>

                <div class="card">
                    <h3>ML Model Performance</h3>
                    <div id="ml-metrics"></div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>CPU Usage</h3>
                    <div class="chart-container">
                        <canvas id="cpu-chart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h3>Memory Usage</h3>
                    <div class="chart-container">
                        <canvas id="memory-chart"></canvas>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Recent API Requests</h3>
                <div id="api-requests-table"></div>
            </div>
        </div>

        <script>
            let cpuChart, memoryChart;

            async function fetchData(endpoint) {
                try {
                    const response = await fetch(endpoint);
                    return await response.json();
                } catch (error) {
                    console.error('Error fetching data:', error);
                    return null;
                }
            }

            async function updateHealthStatus() {
                const health = await fetchData('/admin/monitoring/health');
                if (!health) return;

                const container = document.getElementById('health-status');
                container.innerHTML = '';

                // Overall status
                const overallDiv = document.createElement('div');
                overallDiv.className = `metric-value status-${health.status}`;
                overallDiv.textContent = health.status.toUpperCase();
                container.appendChild(overallDiv);

                // Individual checks
                Object.entries(health.checks).forEach(([name, check]) => {
                    const checkDiv = document.createElement('div');
                    checkDiv.className = `health-check ${check.status}`;
                    checkDiv.innerHTML = `
                        <span>${name}</span>
                        <span class="status-${check.status}">${check.status}</span>
                    `;
                    container.appendChild(checkDiv);
                });
            }

            async function updateMetrics() {
                const summary = await fetchData('/admin/monitoring/metrics/summary');
                if (!summary) return;

                // System metrics
                const systemContainer = document.getElementById('system-metrics');
                systemContainer.innerHTML = '';

                const systemMetrics = [
                    { key: 'system.cpu.usage', label: 'CPU Usage', unit: '%' },
                    { key: 'system.memory.usage', label: 'Memory Usage', unit: '%' },
                    { key: 'system.disk.usage', label: 'Disk Usage', unit: '%' }
                ];

                systemMetrics.forEach(metric => {
                    const value = summary.gauges[metric.key];
                    if (value !== undefined) {
                        const div = document.createElement('div');
                        div.innerHTML = `
                            <div class="metric-value">${value.toFixed(1)}${metric.unit}</div>
                            <div class="metric-label">${metric.label}</div>
                        `;
                        systemContainer.appendChild(div);
                    }
                });

                // API metrics
                const apiContainer = document.getElementById('api-metrics');
                apiContainer.innerHTML = '';

                const apiRequestsCount = Object.entries(summary.counters)
                    .filter(([key]) => key.startsWith('api.requests'))
                    .reduce((sum, [, value]) => sum + value, 0);

                const avgResponseTime = summary.histograms['api.response_time']?.avg || 0;

                apiContainer.innerHTML = `
                    <div class="metric-value">${apiRequestsCount}</div>
                    <div class="metric-label">Total Requests</div>
                    <div class="metric-value">${avgResponseTime.toFixed(2)}ms</div>
                    <div class="metric-label">Avg Response Time</div>
                `;

                // ML metrics
                const mlContainer = document.getElementById('ml-metrics');
                mlContainer.innerHTML = '';

                const mlPredictions = Object.entries(summary.counters)
                    .filter(([key]) => key.startsWith('ml.predictions'))
                    .reduce((sum, [, value]) => sum + value, 0);

                const avgPredictionTime = summary.histograms['ml.prediction_time']?.avg || 0;

                mlContainer.innerHTML = `
                    <div class="metric-value">${mlPredictions}</div>
                    <div class="metric-label">Total Predictions</div>
                    <div class="metric-value">${avgPredictionTime.toFixed(2)}ms</div>
                    <div class="metric-label">Avg Prediction Time</div>
                `;
            }

            async function updateCharts() {
                const metrics = await fetchData('/admin/monitoring/metrics?since_minutes=60');
                if (!metrics) return;

                // CPU Chart
                if (metrics['system.cpu.usage']) {
                    const cpuData = metrics['system.cpu.usage'];
                    const labels = cpuData.map(point => new Date(point.timestamp).toLocaleTimeString());
                    const values = cpuData.map(point => point.value);

                    if (cpuChart) {
                        cpuChart.data.labels = labels;
                        cpuChart.data.datasets[0].data = values;
                        cpuChart.update();
                    } else {
                        const ctx = document.getElementById('cpu-chart').getContext('2d');
                        cpuChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'CPU Usage (%)',
                                    data: values,
                                    borderColor: '#2563eb',
                                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                                    tension: 0.4
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
                                    }
                                }
                            }
                        });
                    }
                }

                // Memory Chart
                if (metrics['system.memory.usage']) {
                    const memoryData = metrics['system.memory.usage'];
                    const labels = memoryData.map(point => new Date(point.timestamp).toLocaleTimeString());
                    const values = memoryData.map(point => point.value);

                    if (memoryChart) {
                        memoryChart.data.labels = labels;
                        memoryChart.data.datasets[0].data = values;
                        memoryChart.update();
                    } else {
                        const ctx = document.getElementById('memory-chart').getContext('2d');
                        memoryChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Memory Usage (%)',
                                    data: values,
                                    borderColor: '#10b981',
                                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                    tension: 0.4
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
                                    }
                                }
                            }
                        });
                    }
                }
            }

            async function refreshData() {
                document.getElementById('last-updated').textContent = 'Updating...';
                
                await Promise.all([
                    updateHealthStatus(),
                    updateMetrics(),
                    updateCharts()
                ]);

                document.getElementById('last-updated').textContent = 
                    `Last updated: ${new Date().toLocaleTimeString()}`;
            }

            // Initial load and auto-refresh
            refreshData();
            setInterval(refreshData, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    
    return dashboard_html


@router.get("/alerts")
async def get_recent_alerts(
    hours: int = Query(24, description="Hours of alert history"),
    current_user = Depends(get_current_admin_user)
):
    """Get recent system alerts."""
    # This would typically query a database or log store
    # For now, return a placeholder response
    return {
        "alerts": [],
        "message": "Alert history feature coming soon"
    }


@router.post("/alerts/test")
async def test_alert_system(current_user = Depends(get_current_admin_user)):
    """Test the alert system."""
    test_alert = {
        "check_name": "test_check",
        "current_status": "critical",
        "current_message": "This is a test alert",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Send test alert to all handlers
    for handler in health_monitor.alert_handlers:
        try:
            if hasattr(handler, '__call__'):
                await handler(test_alert)
        except Exception as e:
            logger.error(f"Error sending test alert: {e}")
    
    return {"message": "Test alert sent successfully"}