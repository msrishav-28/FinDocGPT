#!/bin/bash

# Environment Setup Script for Financial Intelligence System
# Sets up development and production environments with all necessary dependencies

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
ENVIRONMENT=${ENVIRONMENT:-development}
SKIP_DOCKER=${SKIP_DOCKER:-false}
SKIP_NODE=${SKIP_NODE:-false}
SKIP_PYTHON=${SKIP_PYTHON:-false}

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

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_success "Operating system detected: $OS"
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 4 ]; then
            log_warning "System has less than 4GB RAM. Performance may be affected."
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 5000000 ]; then  # 5GB in KB
        log_warning "Less than 5GB disk space available. Consider freeing up space."
    fi
}

setup_environment_file() {
    log_info "Setting up environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Create .env file from template if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp ".env.example" ".env"
            log_success "Created .env file from template"
        else
            log_error ".env.example file not found"
            exit 1
        fi
    else
        log_info ".env file already exists"
    fi
    
    # Set environment-specific defaults
    case $ENVIRONMENT in
        development)
            sed -i.bak 's/ENVIRONMENT=.*/ENVIRONMENT=development/' .env
            sed -i.bak 's/DEBUG=.*/DEBUG=true/' .env
            sed -i.bak 's/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/' .env
            ;;
        staging)
            sed -i.bak 's/ENVIRONMENT=.*/ENVIRONMENT=staging/' .env
            sed -i.bak 's/DEBUG=.*/DEBUG=false/' .env
            sed -i.bak 's/LOG_LEVEL=.*/LOG_LEVEL=INFO/' .env
            ;;
        production)
            sed -i.bak 's/ENVIRONMENT=.*/ENVIRONMENT=production/' .env
            sed -i.bak 's/DEBUG=.*/DEBUG=false/' .env
            sed -i.bak 's/LOG_LEVEL=.*/LOG_LEVEL=WARNING/' .env
            ;;
    esac
    
    # Clean up backup file
    rm -f .env.bak
    
    log_success "Environment configuration updated for $ENVIRONMENT"
}

install_docker() {
    if [ "$SKIP_DOCKER" = "true" ]; then
        log_info "Skipping Docker installation"
        return
    fi
    
    log_info "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker is already installed (version $DOCKER_VERSION)"
    else
        log_info "Installing Docker..."
        
        case $OS in
            linux)
                # Install Docker on Linux
                curl -fsSL https://get.docker.com -o get-docker.sh
                sudo sh get-docker.sh
                sudo usermod -aG docker $USER
                rm get-docker.sh
                ;;
            macos)
                log_warning "Please install Docker Desktop for Mac from https://docker.com/products/docker-desktop"
                log_warning "After installation, restart this script"
                exit 1
                ;;
            windows)
                log_warning "Please install Docker Desktop for Windows from https://docker.com/products/docker-desktop"
                log_warning "After installation, restart this script"
                exit 1
                ;;
        esac
        
        log_success "Docker installed successfully"
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker Compose is available (version $COMPOSE_VERSION)"
    elif docker compose version &> /dev/null; then
        log_success "Docker Compose (plugin) is available"
    else
        log_error "Docker Compose is not available"
        exit 1
    fi
}

setup_python_environment() {
    if [ "$SKIP_PYTHON" = "true" ]; then
        log_info "Skipping Python environment setup"
        return
    fi
    
    log_info "Setting up Python environment..."
    
    cd "$PROJECT_ROOT/backend"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python 3 is available (version $PYTHON_VERSION)"
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements based on environment
    case $ENVIRONMENT in
        development)
            log_info "Installing development dependencies..."
            pip install -r requirements/development.txt
            ;;
        staging|production)
            log_info "Installing production dependencies..."
            pip install -r requirements/production.txt
            ;;
    esac
    
    log_success "Python environment setup completed"
}

setup_node_environment() {
    if [ "$SKIP_NODE" = "true" ]; then
        log_info "Skipping Node.js environment setup"
        return
    fi
    
    log_info "Setting up Node.js environment..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # Check Node.js version
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js is available (version $NODE_VERSION)"
    else
        log_error "Node.js is not installed"
        log_info "Please install Node.js from https://nodejs.org/"
        exit 1
    fi
    
    # Check npm version
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        log_success "npm is available (version $NPM_VERSION)"
    else
        log_error "npm is not available"
        exit 1
    fi
    
    # Install dependencies
    log_info "Installing Node.js dependencies..."
    npm install
    
    log_success "Node.js environment setup completed"
}

setup_database() {
    log_info "Setting up database..."
    
    cd "$PROJECT_ROOT"
    
    # Check if database is running via Docker
    if docker ps | grep -q postgres; then
        log_success "PostgreSQL is already running in Docker"
    else
        log_info "Starting PostgreSQL with Docker Compose..."
        docker-compose -f infrastructure/docker/docker-compose.yml up -d postgres
        
        # Wait for database to be ready
        log_info "Waiting for database to be ready..."
        sleep 10
        
        # Test database connection
        for i in {1..30}; do
            if docker-compose -f infrastructure/docker/docker-compose.yml exec -T postgres pg_isready -U postgres &> /dev/null; then
                log_success "Database is ready"
                break
            fi
            
            if [ $i -eq 30 ]; then
                log_error "Database failed to start after 30 attempts"
                exit 1
            fi
            
            sleep 2
        done
    fi
    
    # Run database migrations
    log_info "Running database migrations..."
    cd backend
    source venv/bin/activate
    python setup_database.py
    
    log_success "Database setup completed"
}

setup_redis() {
    log_info "Setting up Redis..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Redis is running via Docker
    if docker ps | grep -q redis; then
        log_success "Redis is already running in Docker"
    else
        log_info "Starting Redis with Docker Compose..."
        docker-compose -f infrastructure/docker/docker-compose.yml up -d redis
        
        # Wait for Redis to be ready
        log_info "Waiting for Redis to be ready..."
        sleep 5
        
        # Test Redis connection
        for i in {1..10}; do
            if docker-compose -f infrastructure/docker/docker-compose.yml exec -T redis redis-cli ping | grep -q PONG; then
                log_success "Redis is ready"
                break
            fi
            
            if [ $i -eq 10 ]; then
                log_error "Redis failed to start after 10 attempts"
                exit 1
            fi
            
            sleep 2
        done
    fi
}

create_directories() {
    log_info "Creating necessary directories..."
    
    cd "$PROJECT_ROOT"
    
    # Create directories that might not exist
    mkdir -p logs
    mkdir -p uploads
    mkdir -p temp
    mkdir -p backups
    
    # Set appropriate permissions
    chmod 755 logs uploads temp backups
    
    log_success "Directories created successfully"
}

run_initial_tests() {
    log_info "Running initial tests to verify setup..."
    
    # Test backend
    cd "$PROJECT_ROOT/backend"
    source venv/bin/activate
    
    # Run a simple import test
    if python -c "import app.main" &> /dev/null; then
        log_success "Backend imports successfully"
    else
        log_error "Backend import test failed"
        exit 1
    fi
    
    # Test frontend build
    cd "$PROJECT_ROOT/frontend"
    if npm run build &> /dev/null; then
        log_success "Frontend builds successfully"
    else
        log_warning "Frontend build test failed, but continuing..."
    fi
    
    log_success "Initial tests completed"
}

show_setup_summary() {
    log_info "Setup Summary:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Project Root: $PROJECT_ROOT"
    echo ""
    
    log_info "Next Steps:"
    echo "  1. Review and update .env file with your specific configuration"
    echo "  2. Start the development environment:"
    echo "     cd $PROJECT_ROOT"
    echo "     ./scripts/deploy.sh --type docker-compose --environment $ENVIRONMENT"
    echo "  3. Access the application:"
    echo "     Frontend: http://localhost:5173"
    echo "     Backend API: http://localhost:8000"
    echo "     API Documentation: http://localhost:8000/docs"
    echo ""
    
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "Development Tools:"
        echo "  - Backend virtual environment: source backend/venv/bin/activate"
        echo "  - Frontend development server: cd frontend && npm run dev"
        echo "  - Database management: Access via pgAdmin or psql"
        echo "  - Celery monitoring: http://localhost:5555 (Flower)"
    fi
}

# Main execution
main() {
    log_info "Starting environment setup for Financial Intelligence System..."
    log_info "Environment: $ENVIRONMENT"
    
    check_system_requirements
    setup_environment_file
    
    if [ "$ENVIRONMENT" = "development" ]; then
        install_docker
        setup_python_environment
        setup_node_environment
        create_directories
        setup_database
        setup_redis
        run_initial_tests
    else
        install_docker
        setup_python_environment
        create_directories
    fi
    
    show_setup_summary
    
    log_success "Environment setup completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-node)
            SKIP_NODE=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment ENV    Environment (development|staging|production)"
            echo "  --skip-docker           Skip Docker installation"
            echo "  --skip-node             Skip Node.js environment setup"
            echo "  --skip-python           Skip Python environment setup"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Setup development environment"
            echo "  $0 -e production                     # Setup production environment"
            echo "  $0 --skip-docker --skip-node         # Setup only Python environment"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    log_error "Invalid environment: $ENVIRONMENT"
    log_error "Supported environments: development, staging, production"
    exit 1
fi

# Run main function
main