#!/bin/bash

# Cleanup Script for Financial Intelligence System
# Performs maintenance tasks including cache cleanup, log rotation, and resource optimization

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
CLEANUP_TYPE=${CLEANUP_TYPE:-standard}
KEEP_LOGS_DAYS=${KEEP_LOGS_DAYS:-7}
KEEP_BACKUPS_DAYS=${KEEP_BACKUPS_DAYS:-30}
DRY_RUN=${DRY_RUN:-false}

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

execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would execute: $description"
        log_info "[DRY RUN] Command: $cmd"
    else
        log_info "Executing: $description"
        eval "$cmd"
    fi
}

cleanup_python_cache() {
    log_info "Cleaning up Python cache files..."
    
    cd "$PROJECT_ROOT"
    
    # Find and remove __pycache__ directories
    PYCACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
    if [ "$PYCACHE_DIRS" -gt 0 ]; then
        execute_command "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true" "Remove __pycache__ directories"
        log_success "Removed $PYCACHE_DIRS __pycache__ directories"
    else
        log_info "No __pycache__ directories found"
    fi
    
    # Find and remove .pyc files
    PYC_FILES=$(find . -name "*.pyc" 2>/dev/null | wc -l)
    if [ "$PYC_FILES" -gt 0 ]; then
        execute_command "find . -name '*.pyc' -delete" "Remove .pyc files"
        log_success "Removed $PYC_FILES .pyc files"
    else
        log_info "No .pyc files found"
    fi
    
    # Find and remove .pyo files
    PYO_FILES=$(find . -name "*.pyo" 2>/dev/null | wc -l)
    if [ "$PYO_FILES" -gt 0 ]; then
        execute_command "find . -name '*.pyo' -delete" "Remove .pyo files"
        log_success "Removed $PYO_FILES .pyo files"
    else
        log_info "No .pyo files found"
    fi
    
    # Clean pytest cache
    if [ -d ".pytest_cache" ]; then
        execute_command "rm -rf .pytest_cache" "Remove pytest cache"
        log_success "Removed pytest cache"
    fi
}

cleanup_node_cache() {
    log_info "Cleaning up Node.js cache files..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # Clean npm cache
    if command -v npm &> /dev/null; then
        execute_command "npm cache clean --force" "Clean npm cache"
        log_success "Cleaned npm cache"
    fi
    
    # Remove node_modules if doing deep cleanup
    if [ "$CLEANUP_TYPE" = "deep" ] && [ -d "node_modules" ]; then
        execute_command "rm -rf node_modules" "Remove node_modules directory"
        log_success "Removed node_modules directory"
        log_warning "Run 'npm install' to restore dependencies"
    fi
    
    # Clean build artifacts
    if [ -d "dist" ]; then
        execute_command "rm -rf dist" "Remove build artifacts"
        log_success "Removed build artifacts"
    fi
    
    # Clean coverage reports
    if [ -d "coverage" ]; then
        execute_command "rm -rf coverage" "Remove coverage reports"
        log_success "Removed coverage reports"
    fi
}

cleanup_docker_resources() {
    log_info "Cleaning up Docker resources..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found, skipping Docker cleanup"
        return
    fi
    
    # Remove stopped containers
    STOPPED_CONTAINERS=$(docker ps -aq --filter "status=exited" 2>/dev/null | wc -l)
    if [ "$STOPPED_CONTAINERS" -gt 0 ]; then
        execute_command "docker container prune -f" "Remove stopped containers"
        log_success "Removed $STOPPED_CONTAINERS stopped containers"
    else
        log_info "No stopped containers found"
    fi
    
    # Remove unused images
    if [ "$CLEANUP_TYPE" = "deep" ]; then
        execute_command "docker image prune -a -f" "Remove all unused images"
        log_success "Removed unused Docker images"
    else
        execute_command "docker image prune -f" "Remove dangling images"
        log_success "Removed dangling Docker images"
    fi
    
    # Remove unused volumes
    UNUSED_VOLUMES=$(docker volume ls -qf dangling=true 2>/dev/null | wc -l)
    if [ "$UNUSED_VOLUMES" -gt 0 ]; then
        execute_command "docker volume prune -f" "Remove unused volumes"
        log_success "Removed $UNUSED_VOLUMES unused volumes"
    else
        log_info "No unused volumes found"
    fi
    
    # Remove unused networks
    execute_command "docker network prune -f" "Remove unused networks"
    log_success "Cleaned up unused networks"
    
    # Clean build cache
    if [ "$CLEANUP_TYPE" = "deep" ]; then
        execute_command "docker builder prune -a -f" "Remove build cache"
        log_success "Removed Docker build cache"
    fi
}

cleanup_logs() {
    log_info "Cleaning up log files..."
    
    cd "$PROJECT_ROOT"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Find and rotate/remove old log files
    if [ -d "logs" ]; then
        # Remove logs older than specified days
        OLD_LOGS=$(find logs -name "*.log" -type f -mtime +$KEEP_LOGS_DAYS 2>/dev/null | wc -l)
        if [ "$OLD_LOGS" -gt 0 ]; then
            execute_command "find logs -name '*.log' -type f -mtime +$KEEP_LOGS_DAYS -delete" "Remove logs older than $KEEP_LOGS_DAYS days"
            log_success "Removed $OLD_LOGS old log files"
        else
            log_info "No old log files found"
        fi
        
        # Compress large current log files
        LARGE_LOGS=$(find logs -name "*.log" -type f -size +100M 2>/dev/null)
        if [ -n "$LARGE_LOGS" ]; then
            for log_file in $LARGE_LOGS; do
                if command -v gzip &> /dev/null; then
                    execute_command "gzip '$log_file'" "Compress large log file: $log_file"
                    log_success "Compressed $log_file"
                fi
            done
        fi
    fi
    
    # Clean Docker logs if containers are running
    if command -v docker &> /dev/null; then
        RUNNING_CONTAINERS=$(docker ps -q)
        for container in $RUNNING_CONTAINERS; do
            CONTAINER_NAME=$(docker inspect --format='{{.Name}}' "$container" | sed 's/\///')
            LOG_SIZE=$(docker logs --details "$container" 2>/dev/null | wc -c)
            if [ "$LOG_SIZE" -gt 104857600 ]; then  # 100MB
                log_warning "Container $CONTAINER_NAME has large logs ($(($LOG_SIZE / 1024 / 1024))MB)"
                log_info "Consider restarting the container to clear logs"
            fi
        done
    fi
}

cleanup_temporary_files() {
    log_info "Cleaning up temporary files..."
    
    cd "$PROJECT_ROOT"
    
    # Remove temporary directories
    TEMP_DIRS=("temp" "tmp" ".tmp")
    for dir in "${TEMP_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            TEMP_FILES=$(find "$dir" -type f 2>/dev/null | wc -l)
            if [ "$TEMP_FILES" -gt 0 ]; then
                execute_command "find '$dir' -type f -delete" "Remove temporary files from $dir"
                log_success "Removed $TEMP_FILES temporary files from $dir"
            fi
        fi
    done
    
    # Remove editor backup files
    BACKUP_FILES=$(find . -name "*.bak" -o -name "*.swp" -o -name "*.swo" -o -name "*~" 2>/dev/null | wc -l)
    if [ "$BACKUP_FILES" -gt 0 ]; then
        execute_command "find . -name '*.bak' -o -name '*.swp' -o -name '*.swo' -o -name '*~' -delete" "Remove editor backup files"
        log_success "Removed $BACKUP_FILES editor backup files"
    else
        log_info "No editor backup files found"
    fi
    
    # Remove OS-specific files
    OS_FILES=$(find . -name ".DS_Store" -o -name "Thumbs.db" -o -name "desktop.ini" 2>/dev/null | wc -l)
    if [ "$OS_FILES" -gt 0 ]; then
        execute_command "find . -name '.DS_Store' -o -name 'Thumbs.db' -o -name 'desktop.ini' -delete" "Remove OS-specific files"
        log_success "Removed $OS_FILES OS-specific files"
    else
        log_info "No OS-specific files found"
    fi
}

cleanup_uploads() {
    log_info "Cleaning up upload directory..."
    
    cd "$PROJECT_ROOT"
    
    if [ -d "uploads" ]; then
        # Remove files older than 30 days from uploads
        OLD_UPLOADS=$(find uploads -type f -mtime +30 2>/dev/null | wc -l)
        if [ "$OLD_UPLOADS" -gt 0 ]; then
            execute_command "find uploads -type f -mtime +30 -delete" "Remove uploads older than 30 days"
            log_success "Removed $OLD_UPLOADS old upload files"
        else
            log_info "No old upload files found"
        fi
        
        # Remove empty directories
        execute_command "find uploads -type d -empty -delete 2>/dev/null || true" "Remove empty upload directories"
    fi
}

cleanup_backups() {
    log_info "Cleaning up backup files..."
    
    cd "$PROJECT_ROOT"
    
    if [ -d "backups" ]; then
        # Remove backups older than specified days
        OLD_BACKUPS=$(find backups -type f -mtime +$KEEP_BACKUPS_DAYS 2>/dev/null | wc -l)
        if [ "$OLD_BACKUPS" -gt 0 ]; then
            execute_command "find backups -type f -mtime +$KEEP_BACKUPS_DAYS -delete" "Remove backups older than $KEEP_BACKUPS_DAYS days"
            log_success "Removed $OLD_BACKUPS old backup files"
        else
            log_info "No old backup files found"
        fi
    fi
}

optimize_database() {
    log_info "Optimizing database..."
    
    cd "$PROJECT_ROOT"
    
    # Check if database is running
    if docker ps | grep -q postgres; then
        log_info "Running database optimization..."
        
        # Run VACUUM and ANALYZE on PostgreSQL
        execute_command "docker-compose -f infrastructure/docker/docker-compose.yml exec -T postgres psql -U postgres -d financial_intelligence -c 'VACUUM ANALYZE;'" "Optimize PostgreSQL database"
        log_success "Database optimization completed"
    else
        log_warning "Database is not running, skipping optimization"
    fi
}

show_disk_usage() {
    log_info "Disk usage summary:"
    
    cd "$PROJECT_ROOT"
    
    # Show directory sizes
    echo "Directory sizes:"
    du -sh . 2>/dev/null || echo "  Unable to calculate total size"
    
    if [ -d "backend" ]; then
        echo "  Backend: $(du -sh backend 2>/dev/null | cut -f1)"
    fi
    
    if [ -d "frontend" ]; then
        echo "  Frontend: $(du -sh frontend 2>/dev/null | cut -f1)"
    fi
    
    if [ -d "logs" ]; then
        echo "  Logs: $(du -sh logs 2>/dev/null | cut -f1)"
    fi
    
    if [ -d "uploads" ]; then
        echo "  Uploads: $(du -sh uploads 2>/dev/null | cut -f1)"
    fi
    
    if [ -d "backups" ]; then
        echo "  Backups: $(du -sh backups 2>/dev/null | cut -f1)"
    fi
    
    # Show Docker usage if available
    if command -v docker &> /dev/null; then
        echo ""
        echo "Docker usage:"
        docker system df 2>/dev/null || echo "  Unable to get Docker usage"
    fi
}

# Main execution
main() {
    log_info "Starting cleanup of Financial Intelligence System..."
    log_info "Cleanup type: $CLEANUP_TYPE"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Show initial disk usage
    show_disk_usage
    echo ""
    
    # Run cleanup tasks
    cleanup_python_cache
    cleanup_node_cache
    cleanup_temporary_files
    cleanup_logs
    cleanup_uploads
    cleanup_backups
    
    if [ "$CLEANUP_TYPE" = "deep" ]; then
        cleanup_docker_resources
        optimize_database
    fi
    
    echo ""
    log_success "Cleanup completed successfully!"
    
    # Show final disk usage
    echo ""
    show_disk_usage
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            CLEANUP_TYPE="$2"
            shift 2
            ;;
        --keep-logs)
            KEEP_LOGS_DAYS="$2"
            shift 2
            ;;
        --keep-backups)
            KEEP_BACKUPS_DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE          Cleanup type (standard|deep)"
            echo "  --keep-logs DAYS         Days to keep log files (default: 7)"
            echo "  --keep-backups DAYS      Days to keep backup files (default: 30)"
            echo "  --dry-run               Show what would be cleaned without making changes"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Cleanup Types:"
            echo "  standard                 Basic cleanup (cache, logs, temp files)"
            echo "  deep                     Deep cleanup (includes Docker resources, database optimization)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Standard cleanup"
            echo "  $0 --type deep                       # Deep cleanup"
            echo "  $0 --dry-run                         # Preview cleanup actions"
            echo "  $0 --keep-logs 14 --keep-backups 60 # Custom retention periods"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate cleanup type
if [ "$CLEANUP_TYPE" != "standard" ] && [ "$CLEANUP_TYPE" != "deep" ]; then
    log_error "Invalid cleanup type: $CLEANUP_TYPE"
    log_error "Supported types: standard, deep"
    exit 1
fi

# Run main function
main