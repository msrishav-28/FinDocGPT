# Deployment Guide

This comprehensive guide covers all deployment options for the Financial Intelligence System, from local development to production-ready Kubernetes deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Security Configuration](#security-configuration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Scaling and Performance](#scaling-and-performance)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements (Development):**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 20GB available space
- Docker 20.10+ and Docker Compose 2.0+

**Recommended Requirements (Production):**
- CPU: 8+ cores
- Memory: 16GB+ RAM
- Storage: 100GB+ SSD
- Kubernetes cluster 1.19+

### Software Dependencies

- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **kubectl**: 1.19+ (for Kubernetes deployment)
- **Git**: For source code management
- **OpenSSL**: For certificate generation

### Network Requirements

- **Ports**: 80, 443 (HTTP/HTTPS), 5432 (PostgreSQL), 6379 (Redis)
- **Outbound Access**: Required for downloading ML models and market data
- **Internal Communication**: Services need to communicate within the cluster

## Development Deployment

### Quick Start with Docker Compose

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd financial-intelligence-system
   cp .env.example .env
   ```

2. **Configure Environment**
   Edit `.env` file with development settings:
   ```bash
   # Database Configuration
   POSTGRES_DB=financial_intelligence
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_PORT=5432
   
   # Redis Configuration
   REDIS_PORT=6379
   
   # Application Configuration
   ENVIRONMENT=development
   ALLOW_MODEL_DOWNLOAD=1
   JWT_SECRET_KEY=dev-secret-key-change-in-production
   LOG_LEVEL=DEBUG
   
   # Service Ports
   BACKEND_PORT=8000
   FRONTEND_PORT=5173
   FLOWER_PORT=5555
   ```

3. **Start Services**
   ```bash
   docker compose up --build
   ```

4. **Verify Deployment**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Flower (Celery Monitor): http://localhost:5555

5. **Stop Services**
   ```bash
   docker compose down
   ```

### Development with Hot Reload

For active development with code changes:

```bash
# Start with development overrides
docker compose -f docker-compose.yml -f docker-compose.override.yml up --build

# Or use watch mode (Docker Compose 2.22+)
docker compose watch
```

### Manual Development Setup

If you prefer running services individually:

**Backend Setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend Setup:**
```bash
cd frontend
npm install
npm run dev
```

**Database Setup:**
```bash
# Start PostgreSQL and Redis
docker compose up postgres redis -d

# Run database migrations
cd backend
python setup_database.py
```

## Production Deployment

### Docker Compose Production

1. **Prepare Production Environment**
   ```bash
   cp .env.example .env.prod
   ```

2. **Configure Production Settings**
   ```bash
   # Security
   ENVIRONMENT=production
   JWT_SECRET_KEY=<generate-strong-secret>
   POSTGRES_PASSWORD=<strong-database-password>
   
   # Performance
   ALLOW_MODEL_DOWNLOAD=0
   LOG_LEVEL=INFO
   
   # Monitoring
   SENTRY_DSN=<your-sentry-dsn>
   FLOWER_USER=admin
   FLOWER_PASSWORD=<secure-password>
   ```

3. **Deploy with Production Configuration**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

4. **SSL/TLS Configuration**
   ```bash
   # Generate SSL certificates
   mkdir -p nginx/ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/private.key \
     -out nginx/ssl/certificate.crt
   
   # Update nginx configuration for HTTPS
   # See nginx/nginx.conf for SSL configuration
   ```

### Production Checklist

- [ ] Strong passwords for all services
- [ ] SSL/TLS certificates configured
- [ ] Environment variables secured
- [ ] Database backups configured
- [ ] Monitoring and alerting setup
- [ ] Log aggregation configured
- [ ] Resource limits set
- [ ] Health checks enabled
- [ ] Security scanning completed

## Kubernetes Deployment

### Cluster Preparation

1. **Verify Cluster Access**
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

2. **Create Required Storage Classes**
   ```yaml
   # storage-classes.yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: fast-ssd
   provisioner: kubernetes.io/gce-pd
   parameters:
     type: pd-ssd
   ---
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: shared-storage
   provisioner: kubernetes.io/nfs
   ```

3. **Apply Storage Classes**
   ```bash
   kubectl apply -f storage-classes.yaml
   ```

### Container Images

1. **Build Backend Image**
   ```bash
   docker build -t your-registry.com/financial-intelligence/backend:latest ./backend
   docker push your-registry.com/financial-intelligence/backend:latest
   ```

2. **Build Frontend Image**
   ```bash
   docker build -t your-registry.com/financial-intelligence/frontend:latest \
     --target production ./frontend
   docker push your-registry.com/financial-intelligence/frontend:latest
   ```

### Automated Deployment

1. **Update Configuration**
   ```bash
   cd k8s
   
   # Update secrets.yaml with production values
   # Update configmap.yaml if needed
   # Update image references in deployment files
   ```

2. **Deploy Using Script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

### Manual Deployment Steps

1. **Create Namespace and Secrets**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/configmap.yaml
   ```

2. **Deploy Database Services**
   ```bash
   kubectl apply -f k8s/postgres.yaml
   kubectl apply -f k8s/redis.yaml
   
   # Wait for databases to be ready
   kubectl wait --for=condition=available --timeout=300s \
     deployment/postgres -n financial-intelligence
   kubectl wait --for=condition=available --timeout=300s \
     deployment/redis -n financial-intelligence
   ```

3. **Deploy Application Services**
   ```bash
   kubectl apply -f k8s/backend.yaml
   kubectl apply -f k8s/celery.yaml
   kubectl apply -f k8s/frontend.yaml
   kubectl apply -f k8s/nginx.yaml
   ```

4. **Verify Deployment**
   ```bash
   kubectl get pods -n financial-intelligence
   kubectl get services -n financial-intelligence
   kubectl get ingress -n financial-intelligence
   ```

### Using Kustomize

For environment-specific configurations:

```bash
cd k8s
kubectl apply -k overlays/production
```

## Environment Configuration

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | development | Yes |
| `POSTGRES_DB` | Database name | financial_intelligence | Yes |
| `POSTGRES_USER` | Database user | postgres | Yes |
| `POSTGRES_PASSWORD` | Database password | postgres | Yes |
| `REDIS_HOST` | Redis hostname | redis | Yes |
| `REDIS_PORT` | Redis port | 6379 | Yes |
| `JWT_SECRET_KEY` | JWT signing key | - | Yes |
| `CELERY_BROKER_URL` | Celery broker URL | - | Yes |
| `CELERY_RESULT_BACKEND` | Celery result backend | - | Yes |
| `LOG_LEVEL` | Logging level | INFO | No |
| `SENTRY_DSN` | Error tracking DSN | - | No |
| `ALLOW_MODEL_DOWNLOAD` | Allow ML model downloads | 0 | No |

### Configuration Templates

**Development (.env.dev):**
```bash
ENVIRONMENT=development
POSTGRES_PASSWORD=postgres
JWT_SECRET_KEY=dev-secret-key
ALLOW_MODEL_DOWNLOAD=1
LOG_LEVEL=DEBUG
```

**Production (.env.prod):**
```bash
ENVIRONMENT=production
POSTGRES_PASSWORD=<secure-password>
JWT_SECRET_KEY=<strong-secret-key>
ALLOW_MODEL_DOWNLOAD=0
LOG_LEVEL=INFO
SENTRY_DSN=<your-sentry-dsn>
```

## Security Configuration

### SSL/TLS Setup

1. **Generate Certificates**
   ```bash
   # Self-signed certificate (development)
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout tls.key -out tls.crt
   
   # Let's Encrypt (production)
   certbot certonly --standalone -d your-domain.com
   ```

2. **Configure Nginx**
   ```nginx
   server {
       listen 443 ssl http2;
       server_name your-domain.com;
       
       ssl_certificate /etc/nginx/ssl/tls.crt;
       ssl_certificate_key /etc/nginx/ssl/tls.key;
       
       # SSL configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
   }
   ```

### Network Security

1. **Kubernetes Network Policies**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: financial-intelligence-netpol
     namespace: financial-intelligence
   spec:
     podSelector: {}
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from:
       - namespaceSelector:
           matchLabels:
             name: financial-intelligence
   ```

2. **Database Security**
   ```bash
   # Restrict database access
   kubectl patch service postgres-service -p '{"spec":{"type":"ClusterIP"}}'
   
   # Use strong passwords
   kubectl create secret generic db-secret \
     --from-literal=password=$(openssl rand -base64 32)
   ```

### Secrets Management

1. **Kubernetes Secrets**
   ```bash
   # Create secrets from files
   kubectl create secret generic app-secrets \
     --from-file=jwt-secret=jwt.key \
     --from-file=db-password=db.pass \
     -n financial-intelligence
   ```

2. **External Secrets Operator**
   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: SecretStore
   metadata:
     name: vault-backend
   spec:
     provider:
       vault:
         server: "https://vault.example.com"
         path: "secret"
   ```

## Monitoring and Logging

### Health Checks

All services include comprehensive health checks:

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:5173/health

# Kubernetes health checks
kubectl get pods -n financial-intelligence
kubectl describe pod <pod-name> -n financial-intelligence
```

### Logging Configuration

1. **Centralized Logging**
   ```yaml
   # fluentd-config.yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: fluentd-config
   data:
     fluent.conf: |
       <source>
         @type tail
         path /var/log/containers/*.log
         pos_file /var/log/fluentd-containers.log.pos
         tag kubernetes.*
         format json
       </source>
   ```

2. **Log Aggregation**
   ```bash
   # Deploy ELK stack
   kubectl apply -f monitoring/elasticsearch.yaml
   kubectl apply -f monitoring/logstash.yaml
   kubectl apply -f monitoring/kibana.yaml
   ```

### Monitoring Stack

1. **Prometheus and Grafana**
   ```bash
   # Deploy monitoring stack
   kubectl apply -f monitoring/prometheus/
   kubectl apply -f monitoring/grafana/
   ```

2. **Custom Metrics**
   ```python
   # Backend metrics endpoint
   from prometheus_client import Counter, Histogram
   
   REQUEST_COUNT = Counter('requests_total', 'Total requests')
   REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
   ```

## Scaling and Performance

### Horizontal Pod Autoscaling

1. **CPU-based Scaling**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: backend-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: backend
     minReplicas: 3
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

2. **Memory-based Scaling**
   ```yaml
   metrics:
   - type: Resource
     resource:
       name: memory
       target:
         type: Utilization
         averageUtilization: 80
   ```

### Manual Scaling

```bash
# Scale backend pods
kubectl scale deployment backend --replicas=5 -n financial-intelligence

# Scale Celery workers
kubectl scale deployment celery-worker --replicas=8 -n financial-intelligence

# Scale frontend pods
kubectl scale deployment frontend --replicas=3 -n financial-intelligence
```

### Performance Optimization

1. **Database Optimization**
   ```sql
   -- PostgreSQL configuration
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   maintenance_work_mem = 64MB
   ```

2. **Redis Configuration**
   ```redis
   # redis.conf
   maxmemory 512mb
   maxmemory-policy allkeys-lru
   save 900 1
   save 300 10
   ```

3. **Application Tuning**
   ```python
   # FastAPI configuration
   app = FastAPI(
       title="Financial Intelligence API",
       docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
       redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None
   )
   ```

## Backup and Recovery

### Database Backup

1. **Automated Backups**
   ```bash
   # Create backup script
   #!/bin/bash
   BACKUP_DIR="/backups"
   DATE=$(date +%Y%m%d_%H%M%S)
   
   kubectl exec deployment/postgres -n financial-intelligence -- \
     pg_dump -U postgres financial_intelligence > \
     $BACKUP_DIR/backup_$DATE.sql
   ```

2. **Backup CronJob**
   ```yaml
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: postgres-backup
   spec:
     schedule: "0 2 * * *"  # Daily at 2 AM
     jobTemplate:
       spec:
         template:
           spec:
             containers:
             - name: postgres-backup
               image: postgres:15-alpine
               command:
               - /bin/bash
               - -c
               - pg_dump -h postgres-service -U postgres financial_intelligence > /backup/backup_$(date +%Y%m%d_%H%M%S).sql
   ```

### Disaster Recovery

1. **Database Restore**
   ```bash
   # Restore from backup
   kubectl exec -i deployment/postgres -n financial-intelligence -- \
     psql -U postgres financial_intelligence < backup.sql
   ```

2. **Application State Recovery**
   ```bash
   # Restore persistent volumes
   kubectl apply -f backup-pvc.yaml
   
   # Restore application data
   kubectl cp backup-data.tar.gz pod-name:/app/data/
   ```

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   # Check pod status
   kubectl describe pod <pod-name> -n financial-intelligence
   
   # Check logs
   kubectl logs <pod-name> -n financial-intelligence
   
   # Check events
   kubectl get events -n financial-intelligence --sort-by='.lastTimestamp'
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -it deployment/postgres -n financial-intelligence -- \
     psql -U postgres -d financial_intelligence -c "SELECT 1;"
   
   # Check service endpoints
   kubectl get endpoints postgres-service -n financial-intelligence
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connectivity
   kubectl exec -it deployment/redis -n financial-intelligence -- \
     redis-cli ping
   
   # Check Redis logs
   kubectl logs deployment/redis -n financial-intelligence
   ```

4. **Image Pull Errors**
   ```bash
   # Check image pull secrets
   kubectl get secrets -n financial-intelligence
   
   # Verify registry credentials
   kubectl describe secret registry-secret -n financial-intelligence
   ```

### Performance Issues

1. **High CPU Usage**
   ```bash
   # Check resource usage
   kubectl top pods -n financial-intelligence
   
   # Scale up if needed
   kubectl scale deployment backend --replicas=5 -n financial-intelligence
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   kubectl describe node <node-name>
   
   # Adjust resource limits
   kubectl patch deployment backend -p '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
   ```

3. **Database Performance**
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Check database connections
   SELECT count(*) FROM pg_stat_activity;
   ```

### Network Issues

1. **Service Discovery**
   ```bash
   # Check DNS resolution
   kubectl exec -it deployment/backend -n financial-intelligence -- \
     nslookup postgres-service
   
   # Check service endpoints
   kubectl get endpoints -n financial-intelligence
   ```

2. **Ingress Issues**
   ```bash
   # Check ingress status
   kubectl describe ingress -n financial-intelligence
   
   # Check ingress controller logs
   kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
   ```

### Recovery Procedures

1. **Rolling Restart**
   ```bash
   # Restart all deployments
   kubectl rollout restart deployment -n financial-intelligence
   ```

2. **Rollback Deployment**
   ```bash
   # Check rollout history
   kubectl rollout history deployment/backend -n financial-intelligence
   
   # Rollback to previous version
   kubectl rollout undo deployment/backend -n financial-intelligence
   ```

3. **Emergency Procedures**
   ```bash
   # Scale down problematic services
   kubectl scale deployment backend --replicas=0 -n financial-intelligence
   
   # Drain node for maintenance
   kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
   ```

## Support and Maintenance

### Regular Maintenance Tasks

- **Weekly**: Check logs for errors and warnings
- **Monthly**: Update container images and security patches
- **Quarterly**: Review and update SSL certificates
- **Annually**: Conduct security audits and penetration testing

### Monitoring Checklist

- [ ] All pods are running and healthy
- [ ] Database connections are stable
- [ ] Redis cache is functioning
- [ ] SSL certificates are valid
- [ ] Backup jobs are completing successfully
- [ ] Resource usage is within acceptable limits
- [ ] No critical security vulnerabilities

### Getting Help

For deployment issues:
1. Check the troubleshooting section above
2. Review logs for error messages
3. Verify configuration values
4. Ensure all prerequisites are met
5. Check resource availability and limits
6. Review security policies and network connectivity

For additional support, please refer to the project documentation or contact the development team.