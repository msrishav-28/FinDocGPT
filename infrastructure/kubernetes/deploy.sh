#!/bin/bash

# Kubernetes deployment script for Financial Intelligence System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="financial-intelligence"
REGISTRY="your-registry.com"
IMAGE_TAG=${IMAGE_TAG:-latest}

echo -e "${GREEN}Starting deployment of Financial Intelligence System...${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Not connected to a Kubernetes cluster${NC}"
    exit 1
fi

# Function to wait for deployment
wait_for_deployment() {
    local deployment=$1
    echo -e "${YELLOW}Waiting for deployment $deployment to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE
}

# Function to wait for pods
wait_for_pods() {
    local label=$1
    echo -e "${YELLOW}Waiting for pods with label $label to be ready...${NC}"
    kubectl wait --for=condition=ready --timeout=300s pod -l $label -n $NAMESPACE
}

# Create namespace
echo -e "${YELLOW}Creating namespace...${NC}"
kubectl apply -f namespace.yaml

# Apply secrets (make sure to update with real values)
echo -e "${YELLOW}Applying secrets...${NC}"
kubectl apply -f secrets.yaml

# Apply configmaps
echo -e "${YELLOW}Applying configmaps...${NC}"
kubectl apply -f configmap.yaml

# Deploy database services
echo -e "${YELLOW}Deploying PostgreSQL...${NC}"
kubectl apply -f postgres.yaml
wait_for_deployment postgres

echo -e "${YELLOW}Deploying Redis...${NC}"
kubectl apply -f redis.yaml
wait_for_deployment redis

# Deploy application services
echo -e "${YELLOW}Deploying backend...${NC}"
kubectl apply -f backend.yaml
wait_for_deployment backend

echo -e "${YELLOW}Deploying Celery workers...${NC}"
kubectl apply -f celery.yaml
wait_for_deployment celery-worker
wait_for_deployment celery-beat

echo -e "${YELLOW}Deploying frontend...${NC}"
kubectl apply -f frontend.yaml
wait_for_deployment frontend

# Deploy load balancer
echo -e "${YELLOW}Deploying nginx load balancer...${NC}"
kubectl apply -f nginx.yaml
wait_for_deployment nginx

# Check deployment status
echo -e "${GREEN}Deployment completed! Checking status...${NC}"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Get external IP
echo -e "${GREEN}Getting external access information...${NC}"
kubectl get service nginx-service -n $NAMESPACE

echo -e "${GREEN}Financial Intelligence System deployed successfully!${NC}"
echo -e "${YELLOW}Note: Make sure to update secrets.yaml with real production values before deploying to production.${NC}"