#!/bin/bash

# Deployment script for Financial Intelligence System
# Supports both Docker Compose and Kubernetes deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-docker-compose}
ENVIRONMENT=${ENVIRONMENT:-development}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-ghcr.io/your-org/financial-intelligence}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed or not in PATH"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Not connected to a Kubernetes cluster"
            exit 1
        fi
    elif [ "$DEPLOYMENT_TYPE" = "docker-compose" ]; then
        if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
            log_error "Docker Compose is not installed or not in PATH"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment for $ENVIRONMENT..."
    
    # Create environment file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            log_warning "Created .env file from .env.example. Please review and update the configuration."
        else
            log_error ".env.example file not found"
            exit 1
        fi
    fi
    
    # Set environment-specific variables
    case $ENVIRONMENT in
        development)
            export FRONTEND_TARGET=development
            export FRONTEND_INTERNAL_PORT=5173
            ;;
        staging|production)
            export FRONTEND_TARGET=production
            export FRONTEND_INTERNAL_PORT=80
            ;;
    esac
    
    log_success "Environment setup completed"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Choose the appropriate compose file based on new structure
    COMPOSE_FILE="infrastructure/docker/docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ] && [ -f "infrastructure/docker/docker-compose.prod.yml" ]; then
        COMPOSE_FILE="$COMPOSE_FILE -f infrastructure/docker/docker-compose.prod.yml"
    fi
    
    # Pull latest images if using remote registry
    if [ "$IMAGE_TAG" != "latest" ] || [ -n "$REGISTRY" ]; then
        log_info "Pulling latest images..."
        docker-compose -f $COMPOSE_FILE pull
    fi
    
    # Build and start services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d --build
    
    # Wait for services to be healthy
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if docker-compose -f $COMPOSE_FILE ps | grep -q "unhealthy\|Exit"; then
        log_error "Some services are not healthy"
        docker-compose -f $COMPOSE_FILE ps
        exit 1
    fi
    
    log_success "Docker Compose deployment completed successfully"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT/infrastructure/kubernetes"
    
    # Update image tags if specified
    if [ "$IMAGE_TAG" != "latest" ]; then
        log_info "Updating image tags to $IMAGE_TAG..."
        if [ -f "backend.yaml" ]; then
            sed -i.bak "s|financial-intelligence/backend:latest|$REGISTRY/backend:$IMAGE_TAG|g" backend.yaml
        fi
        if [ -f "frontend.yaml" ]; then
            sed -i.bak "s|financial-intelligence/frontend:latest|$REGISTRY/frontend:$IMAGE_TAG|g" frontend.yaml
        fi
    fi
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    
    # Create namespace if it doesn't exist
    if [ -f "namespace.yaml" ]; then
        kubectl apply -f namespace.yaml
    fi
    
    # Apply secrets and configmaps
    if [ -f "secrets.yaml" ]; then
        kubectl apply -f secrets.yaml
    fi
    if [ -f "configmap.yaml" ]; then
        kubectl apply -f configmap.yaml
    fi
    
    # Deploy database services first
    if [ -f "postgres.yaml" ]; then
        kubectl apply -f postgres.yaml
        kubectl wait --for=condition=available --timeout=300s deployment/postgres -n financial-intelligence || log_warning "PostgreSQL deployment timeout"
    fi
    if [ -f "redis.yaml" ]; then
        kubectl apply -f redis.yaml
        kubectl wait --for=condition=available --timeout=300s deployment/redis -n financial-intelligence || log_warning "Redis deployment timeout"
    fi
    
    # Deploy application services
    if [ -f "backend.yaml" ]; then
        kubectl apply -f backend.yaml
        kubectl wait --for=condition=available --timeout=600s deployment/backend -n financial-intelligence || log_warning "Backend deployment timeout"
    fi
    if [ -f "celery.yaml" ]; then
        kubectl apply -f celery.yaml
        kubectl wait --for=condition=available --timeout=600s deployment/celery-worker -n financial-intelligence || log_warning "Celery deployment timeout"
    fi
    if [ -f "frontend.yaml" ]; then
        kubectl apply -f frontend.yaml
        kubectl wait --for=condition=available --timeout=600s deployment/frontend -n financial-intelligence || log_warning "Frontend deployment timeout"
    fi
    if [ -f "nginx.yaml" ]; then
        kubectl apply -f nginx.yaml
    fi
    
    # Clean up backup files
    rm -f *.bak
    
    log_success "Kubernetes deployment completed successfully"
}

run_health_checks() {
    log_info "Running health checks..."
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        # Get service URL
        SERVICE_IP=$(kubectl get service nginx-service -n financial-intelligence -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
        SERVICE_PORT=$(kubectl get service nginx-service -n financial-intelligence -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "80")
        BASE_URL="http://$SERVICE_IP:$SERVICE_PORT"
    else
        BASE_URL="http://localhost:8000"
    fi
    
    # Test basic health endpoint
    for i in {1..10}; do
        if curl -f -s "$BASE_URL/health" > /dev/null; then
            log_success "Basic health check passed"
            break
        fi
        
        if [ $i -eq 10 ]; then
            log_error "Health check failed after 10 attempts"
            exit 1
        fi
        
        log_info "Health check attempt $i failed, retrying in 10 seconds..."
        sleep 10
    done
    
    # Test API health endpoint
    if curl -f -s "$BASE_URL/api/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_warning "API health check failed, but basic health is OK"
    fi
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Deployment Type: $DEPLOYMENT_TYPE"
    echo "  Image Tag: $IMAGE_TAG"
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        echo ""
        log_info "Kubernetes Resources:"
        kubectl get pods,services,ingress -n financial-intelligence
        
        # Show external access information
        EXTERNAL_IP=$(kubectl get service nginx-service -n financial-intelligence -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -n "$EXTERNAL_IP" ]; then
            echo ""
            log_success "Application accessible at: http://$EXTERNAL_IP"
        fi
    else
        echo ""
        log_info "Docker Compose Services:"
        docker-compose ps
        
        echo ""
        log_success "Application accessible at:"
        echo "  Frontend: http://localhost:5173"
        echo "  Backend API: http://localhost:8000"
        echo "  API Documentation: http://localhost:8000/docs"
        echo "  Flower (Celery Monitor): http://localhost:5555"
    fi
}

cleanup_on_failure() {
    log_error "Deployment failed, cleaning up..."
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        # Rollback Kubernetes deployment
        kubectl rollout undo deployment/backend -n financial-intelligence --ignore-not-found
        kubectl rollout undo deployment/frontend -n financial-intelligence --ignore-not-found
        kubectl rollout undo deployment/celery-worker -n financial-intelligence --ignore-not-found
    else
        # Stop Docker Compose services
        docker-compose down
    fi
}

# Main execution
main() {
    log_info "Starting deployment of Financial Intelligence System..."
    log_info "Environment: $ENVIRONMENT, Type: $DEPLOYMENT_TYPE"
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    # Run deployment steps
    check_prerequisites
    setup_environment
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    run_health_checks
    show_deployment_info
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE          Deployment type (docker-compose|kubernetes)"
            echo "  -e, --environment ENV    Environment (development|staging|production)"
            echo "  -i, --image-tag TAG      Docker image tag"
            echo "  -r, --registry REGISTRY  Docker registry URL"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate deployment type
if [ "$DEPLOYMENT_TYPE" != "docker-compose" ] && [ "$DEPLOYMENT_TYPE" != "kubernetes" ]; then
    log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    log_error "Supported types: docker-compose, kubernetes"
    exit 1
fi

# Validate environment
if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    log_error "Invalid environment: $ENVIRONMENT"
    log_error "Supported environments: development, staging, production"
    exit 1
fi

# Run main function
main