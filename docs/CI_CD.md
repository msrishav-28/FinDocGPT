# CI/CD Pipeline Guide

This guide covers the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the Financial Intelligence System.

## Overview

The CI/CD pipeline consists of three main workflows:

1. **Continuous Integration (CI)** - Automated testing, linting, and security scanning
2. **Continuous Deployment (CD)** - Automated deployment with blue-green strategy
3. **Model Retraining** - Automated ML model retraining and deployment

## Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Code Push     │───▶│   CI Pipeline   │───▶│   CD Pipeline   │
│   Pull Request  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Security Scan  │    │  Blue-Green     │
                       │  Code Quality   │    │  Deployment     │
                       └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Build Images   │    │  Health Checks  │
                       │  Push Registry  │    │  Monitoring     │
                       └─────────────────┘    └─────────────────┘
```

## Continuous Integration (CI)

### Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Stages

#### 1. Backend Testing
- **Python Setup**: Python 3.11 with dependency caching
- **Linting**: Flake8 for code style and basic error checking
- **Type Checking**: MyPy for static type analysis
- **Validation**: Code structure and import validation
- **Services**: PostgreSQL and Redis for validation

#### 2. Frontend Testing
- **Node.js Setup**: Node.js 18 with npm caching
- **Linting**: ESLint for code quality
- **Type Checking**: TypeScript compiler
- **Validation**: Frontend structure and build validation
- **Build**: Production build verification

#### 3. Security Scanning
- **Vulnerability Scanning**: Trivy for filesystem and dependency scanning
- **Security Linting**: Bandit for Python security issues
- **SARIF Upload**: Results uploaded to GitHub Security tab

#### 4. Image Building
- **Multi-stage Builds**: Optimized Docker images
- **Registry Push**: Images pushed to GitHub Container Registry
- **Caching**: Docker layer caching for faster builds

#### 5. Integration Testing
- **End-to-End Tests**: Full system testing with Docker Compose
- **API Testing**: Comprehensive API endpoint testing
- **Performance Testing**: Basic performance benchmarks

### Configuration

```yaml
# .github/workflows/ci.yml
name: Continuous Integration
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```

## Continuous Deployment (CD)

### Triggers
- Push to `main` branch (automatic staging deployment)
- Git tags starting with `v*` (production deployment)
- Manual workflow dispatch

### Deployment Strategy

#### Blue-Green Deployment
1. **Current Environment Detection**: Identify active environment (blue/green)
2. **New Environment Deployment**: Deploy to inactive environment
3. **Health Checks**: Comprehensive health and performance checks
4. **Traffic Switch**: Route traffic to new environment
5. **Monitoring**: Monitor for issues and rollback if needed
6. **Cleanup**: Remove old environment after successful deployment

#### Staging Deployment
- **Automatic**: Triggered on every push to `main`
- **Environment**: `financial-intelligence-staging` namespace
- **Testing**: Smoke tests and basic functionality verification

#### Production Deployment
- **Manual Approval**: Requires manual approval for production
- **Backup**: Database backup before deployment
- **Monitoring**: Extended monitoring period after deployment
- **Rollback**: Automatic rollback on failure detection

### Configuration

```yaml
# .github/workflows/cd.yml
name: Continuous Deployment
on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  workflow_dispatch:
```

## Model Retraining Pipeline

### Triggers
- **Scheduled**: Daily at 2 AM UTC
- **Manual**: Workflow dispatch with model selection

### Process

#### 1. Performance Check
- **Metrics Collection**: Gather model performance metrics
- **Threshold Comparison**: Compare against performance thresholds
- **Decision**: Determine which models need retraining

#### 2. Model Retraining
- **Data Download**: Fetch latest training data from S3
- **Training**: Retrain models using MLflow for experiment tracking
- **Validation**: Validate new models against test datasets
- **Artifact Storage**: Store model artifacts in S3

#### 3. Model Deployment
- **Kubernetes Job**: Deploy new models using Kubernetes jobs
- **Health Checks**: Verify model endpoints are working
- **Performance Testing**: Test new model performance

### Model Performance Thresholds

| Model Type | Metric | Threshold |
|------------|--------|-----------|
| Sentiment | Accuracy | > 85% |
| Forecast | MAPE | < 15% |
| Anomaly | Precision | > 80% |
| Recommendation | CTR | > 10% |

## Environment Configuration

### Development
- **Deployment**: Docker Compose
- **Database**: Local PostgreSQL
- **Monitoring**: Basic logging
- **Testing**: Full test suite

### Staging
- **Deployment**: Kubernetes
- **Database**: Managed PostgreSQL
- **Monitoring**: Prometheus + Grafana
- **Testing**: Integration and smoke tests

### Production
- **Deployment**: Kubernetes with blue-green
- **Database**: High-availability PostgreSQL
- **Monitoring**: Full observability stack
- **Testing**: Comprehensive health checks

## Security Measures

### Code Security
- **Static Analysis**: Bandit for Python security issues
- **Dependency Scanning**: Trivy for vulnerability detection
- **Secret Management**: GitHub Secrets for sensitive data
- **Image Scanning**: Container image vulnerability scanning

### Deployment Security
- **RBAC**: Kubernetes Role-Based Access Control
- **Network Policies**: Kubernetes network segmentation
- **TLS**: End-to-end encryption
- **Secret Rotation**: Regular rotation of secrets and keys

## Monitoring and Alerting

### CI/CD Monitoring
- **Pipeline Status**: Slack notifications for build status
- **Performance Metrics**: Build time and success rate tracking
- **Error Tracking**: Detailed error reporting and analysis

### Deployment Monitoring
- **Health Checks**: Automated health verification
- **Performance Monitoring**: Response time and error rate tracking
- **Resource Monitoring**: CPU, memory, and disk usage
- **Business Metrics**: Model performance and user engagement

## Rollback Procedures

### Automatic Rollback
- **Health Check Failures**: Automatic rollback on health check failures
- **Performance Degradation**: Rollback on high error rates or slow response times
- **Resource Issues**: Rollback on resource exhaustion

### Manual Rollback
```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend -n financial-intelligence
kubectl rollout undo deployment/frontend -n financial-intelligence

# Docker Compose rollback
docker-compose down
git checkout previous-version
docker-compose up -d
```

## Secrets Management

### Required Secrets

#### GitHub Secrets
- `GITHUB_TOKEN`: GitHub API access
- `KUBE_CONFIG_STAGING`: Kubernetes config for staging
- `KUBE_CONFIG_PRODUCTION`: Kubernetes config for production
- `AWS_ACCESS_KEY_ID`: AWS access for S3 and other services
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `SLACK_WEBHOOK_URL`: Slack notifications
- `MLFLOW_TRACKING_URI`: MLflow server URL

#### Kubernetes Secrets
- `app-secrets`: Application secrets (JWT, database passwords)
- `registry-secret`: Container registry credentials
- `aws-credentials`: AWS credentials for model artifacts

### Secret Rotation
- **Schedule**: Monthly rotation of all secrets
- **Process**: Automated rotation using GitHub Actions
- **Verification**: Automated testing after rotation

## Troubleshooting

### Common Issues

#### CI Pipeline Failures
1. **Test Failures**: Check test logs and fix failing tests
2. **Build Failures**: Verify dependencies and Docker configuration
3. **Security Scan Failures**: Address security vulnerabilities

#### Deployment Failures
1. **Image Pull Errors**: Check registry credentials and image tags
2. **Health Check Failures**: Verify service configuration and dependencies
3. **Resource Issues**: Check cluster resources and scaling policies

#### Model Retraining Issues
1. **Data Access**: Verify S3 credentials and data availability
2. **Training Failures**: Check training data quality and model configuration
3. **Deployment Issues**: Verify model artifact format and deployment scripts

### Debugging Commands

```bash
# Check CI/CD pipeline status
gh workflow list
gh run list --workflow=ci.yml

# Check Kubernetes deployment status
kubectl get pods -n financial-intelligence
kubectl describe deployment backend -n financial-intelligence
kubectl logs -f deployment/backend -n financial-intelligence

# Check Docker Compose status
docker-compose ps
docker-compose logs backend
```

## Performance Optimization

### Build Optimization
- **Docker Layer Caching**: Optimize Dockerfile for better caching
- **Dependency Caching**: Cache pip and npm dependencies
- **Parallel Builds**: Run tests and builds in parallel

### Deployment Optimization
- **Resource Limits**: Set appropriate CPU and memory limits
- **Horizontal Scaling**: Use HPA for automatic scaling
- **Image Optimization**: Use multi-stage builds and minimal base images

## Compliance and Auditing

### Audit Trail
- **Git History**: Complete change history in Git
- **Pipeline Logs**: Detailed logs for all pipeline runs
- **Deployment Records**: Kubernetes events and deployment history

### Compliance Checks
- **Security Scanning**: Regular vulnerability assessments
- **Code Quality**: Automated code quality checks
- **Documentation**: Up-to-date documentation and runbooks

## Best Practices

### Code Quality
- **Linting**: Consistent code style enforcement
- **Testing**: Comprehensive test coverage (>80%)
- **Documentation**: Clear and up-to-date documentation

### Security
- **Least Privilege**: Minimal required permissions
- **Secret Management**: Secure secret storage and rotation
- **Regular Updates**: Keep dependencies and base images updated

### Reliability
- **Health Checks**: Comprehensive health monitoring
- **Rollback Strategy**: Quick and reliable rollback procedures
- **Monitoring**: Proactive monitoring and alerting

## Getting Started

### Prerequisites
1. GitHub repository with appropriate permissions
2. Kubernetes cluster access (for production deployments)
3. Container registry access (GitHub Container Registry)
4. Required secrets configured in GitHub

### Setup Steps
1. **Configure Secrets**: Add all required secrets to GitHub repository
2. **Update Configuration**: Modify workflow files for your environment
3. **Test Pipeline**: Create a test branch and verify CI pipeline
4. **Deploy Staging**: Push to main branch to trigger staging deployment
5. **Deploy Production**: Create a release tag to trigger production deployment

### Monitoring Setup
1. **Configure Alerts**: Set up Slack or email notifications
2. **Dashboard Access**: Ensure access to monitoring dashboards
3. **Log Aggregation**: Configure log collection and analysis

This CI/CD pipeline provides a robust, secure, and scalable deployment process for the Financial Intelligence System, ensuring high quality and reliability in production environments.