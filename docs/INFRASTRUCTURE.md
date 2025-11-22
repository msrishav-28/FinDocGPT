# Infrastructure Configuration

This directory contains all infrastructure-related configurations for the Financial Intelligence System, organized into logical subdirectories for better maintainability and deployment readiness.

## Directory Structure

```
infrastructure/
├── docker/                    # Docker Compose configurations
│   ├── docker-compose.yml     # Development environment
│   └── docker-compose.prod.yml # Production environment
├── kubernetes/                # Kubernetes manifests
│   ├── patches/               # Kustomize patches
│   ├── *.yaml                 # K8s resource definitions
│   ├── kustomization.yaml     # Kustomize configuration
│   └── deploy.sh             # Kubernetes deployment script
└── monitoring/                # Monitoring and observability
    ├── docker-compose.monitoring.yml # Monitoring stack
    ├── prometheus/            # Prometheus configuration
    ├── grafana/              # Grafana dashboards and config
    ├── alertmanager/         # Alert management
    ├── loki/                 # Log aggregation
    └── promtail/             # Log collection
```

## Usage

### Docker Compose Deployment

For development:
```bash
cd infrastructure/docker
docker-compose up -d
```

For production:
```bash
cd infrastructure/docker
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
cd infrastructure/kubernetes
./deploy.sh
```

Or using the main deployment script:
```bash
./scripts/deploy.sh --type kubernetes --environment production
```

### Monitoring Stack

To start the monitoring stack:
```bash
cd infrastructure/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

Access points:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093

## Configuration Files

### Docker Compose
- **docker-compose.yml**: Base configuration for development
- **docker-compose.prod.yml**: Production overrides with optimized settings

### Kubernetes
- **namespace.yaml**: Creates the financial-intelligence namespace
- **configmap.yaml**: Application configuration
- **secrets.yaml**: Sensitive configuration (update with real values)
- **postgres.yaml**: PostgreSQL database deployment
- **redis.yaml**: Redis cache deployment
- **backend.yaml**: FastAPI backend deployment
- **celery.yaml**: Celery worker deployments
- **frontend.yaml**: React frontend deployment
- **nginx.yaml**: Load balancer and ingress

### Monitoring
- **prometheus.yml**: Metrics collection configuration
- **alert_rules.yml**: Alerting rules
- **grafana/**: Dashboard definitions and provisioning
- **loki.yml**: Log aggregation configuration

## Security Notes

- Update `secrets.yaml` with real production values
- Configure proper TLS certificates for production
- Review and adjust resource limits based on your infrastructure
- Ensure proper network policies are in place

## Customization

Each configuration file can be customized for your specific environment:
- Update image repositories and tags
- Adjust resource requests and limits
- Configure persistent storage classes
- Set up proper ingress and load balancing
- Configure monitoring and alerting thresholds