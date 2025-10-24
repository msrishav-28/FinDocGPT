#!/usr/bin/env python3
"""
Comprehensive Project Validation Script

This script provides all validation functionality in a single file:
- Deployment configuration validation
- Docker build validation  
- System validation
- Automated structure tests
"""

import os
import sys
import json
import yaml
import unittest
import subprocess
from pathlib import Path
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectValidator:
    """Comprehensive project validation"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors = []
        self.warnings = []
        self.success_count = 0

    def log_error(self, message: str):
        """Log an error and add to error list"""
        self.errors.append(message)
        logger.error(message)

    def log_warning(self, message: str):
        """Log a warning and add to warning list"""
        self.warnings.append(message)
        logger.warning(message)

    def log_success(self, message: str):
        """Log a success message"""
        self.success_count += 1
        logger.info(f"✓ {message}")

    def validate_project_structure(self) -> bool:
        """Validate overall project structure"""
        logger.info("Validating project structure...")
        
        # Required directories
        required_dirs = [
            "backend", "frontend", "infrastructure", "docs", "scripts"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.log_success(f"Directory '{dir_name}' exists")
            else:
                self.log_error(f"Directory '{dir_name}' missing")
        
        # Required files
        required_files = [
            "README.md", ".gitignore", ".env.example"
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.log_success(f"File '{file_name}' exists")
            else:
                self.log_error(f"File '{file_name}' missing")
        
        return len(self.errors) == 0

    def validate_docker_configurations(self) -> bool:
        """Validate Docker configurations"""
        logger.info("Validating Docker configurations...")
        
        # Check Dockerfiles
        dockerfiles = [
            "backend/Dockerfile",
            "frontend/Dockerfile"
        ]
        
        for dockerfile_path in dockerfiles:
            full_path = self.project_root / dockerfile_path
            if full_path.exists():
                self.log_success(f"Dockerfile '{dockerfile_path}' exists")
                
                # Check content
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                if "FROM" in content and "CMD" in content:
                    self.log_success(f"Dockerfile '{dockerfile_path}' has required instructions")
                else:
                    self.log_error(f"Dockerfile '{dockerfile_path}' missing required instructions")
            else:
                self.log_error(f"Dockerfile '{dockerfile_path}' not found")
        
        # Check Docker Compose files
        compose_files = [
            "infrastructure/docker/docker-compose.yml",
            "infrastructure/docker/docker-compose.prod.yml"
        ]
        
        for compose_file in compose_files:
            full_path = self.project_root / compose_file
            if full_path.exists():
                self.log_success(f"Docker Compose '{compose_file}' exists")
                
                try:
                    with open(full_path, 'r') as f:
                        yaml.safe_load(f)
                    self.log_success(f"Docker Compose '{compose_file}' has valid YAML")
                except yaml.YAMLError:
                    self.log_error(f"Docker Compose '{compose_file}' has invalid YAML")
            else:
                self.log_error(f"Docker Compose '{compose_file}' not found")
        
        return len(self.errors) == 0

    def validate_kubernetes_configurations(self) -> bool:
        """Validate Kubernetes configurations"""
        logger.info("Validating Kubernetes configurations...")
        
        k8s_dir = self.project_root / "infrastructure" / "kubernetes"
        if not k8s_dir.exists():
            self.log_error("Kubernetes directory not found")
            return False
        
        # Check for required files
        required_files = [
            "namespace.yaml", "backend.yaml", "frontend.yaml",
            "postgres.yaml", "redis.yaml", "kustomization.yaml"
        ]
        
        for file_name in required_files:
            file_path = k8s_dir / file_name
            if file_path.exists():
                self.log_success(f"Kubernetes file '{file_name}' exists")
                
                try:
                    with open(file_path, 'r') as f:
                        list(yaml.safe_load_all(f))
                    self.log_success(f"Kubernetes file '{file_name}' has valid YAML")
                except yaml.YAMLError:
                    self.log_error(f"Kubernetes file '{file_name}' has invalid YAML")
            else:
                self.log_error(f"Kubernetes file '{file_name}' not found")
        
        return len(self.errors) == 0

    def validate_backend_structure(self) -> bool:
        """Validate backend structure"""
        logger.info("Validating backend structure...")
        
        backend_path = self.project_root / "backend"
        if not backend_path.exists():
            self.log_error("Backend directory not found")
            return False
        
        # Check required directories
        required_dirs = [
            "app", "app/core", "app/api", "app/services", 
            "app/models", "app/database", "requirements"
        ]
        
        for dir_path in required_dirs:
            full_path = backend_path / dir_path
            if full_path.exists():
                self.log_success(f"Backend directory '{dir_path}' exists")
            else:
                self.log_error(f"Backend directory '{dir_path}' missing")
        
        # Check main application file
        main_file = backend_path / "app" / "main.py"
        if main_file.exists():
            self.log_success("Backend main.py exists")
            
            with open(main_file, 'r') as f:
                content = f.read()
                if "FastAPI" in content:
                    self.log_success("Backend uses FastAPI")
                else:
                    self.log_warning("Backend may not use FastAPI")
        else:
            self.log_error("Backend main.py not found")
        
        return len(self.errors) == 0

    def validate_frontend_structure(self) -> bool:
        """Validate frontend structure"""
        logger.info("Validating frontend structure...")
        
        frontend_path = self.project_root / "frontend"
        if not frontend_path.exists():
            self.log_error("Frontend directory not found")
            return False
        
        # Check required directories
        required_dirs = [
            "src", "src/components", "src/pages", 
            "src/services", "src/hooks", "src/utils"
        ]
        
        for dir_path in required_dirs:
            full_path = frontend_path / dir_path
            if full_path.exists():
                self.log_success(f"Frontend directory '{dir_path}' exists")
            else:
                self.log_warning(f"Frontend directory '{dir_path}' missing (may be optional)")
        
        # Check package.json
        package_json = frontend_path / "package.json"
        if package_json.exists():
            self.log_success("Frontend package.json exists")
            
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                
                if 'react' in package_data.get('dependencies', {}):
                    self.log_success("Frontend uses React")
                else:
                    self.log_warning("Frontend may not use React")
                    
            except json.JSONDecodeError:
                self.log_error("Frontend package.json has invalid JSON")
        else:
            self.log_error("Frontend package.json not found")
        
        return len(self.errors) == 0

    def validate_documentation(self) -> bool:
        """Validate documentation"""
        logger.info("Validating documentation...")
        
        docs_path = self.project_root / "docs"
        if not docs_path.exists():
            self.log_error("Documentation directory not found")
            return False
        
        # Check required documentation files
        required_docs = [
            "DEPLOYMENT.md", "API.md", 
            "ARCHITECTURE.md", "CONTRIBUTING.md"
        ]
        
        for doc_file in required_docs:
            full_path = docs_path / doc_file
            if full_path.exists():
                self.log_success(f"Documentation file '{doc_file}' exists")
                
                # Check content length
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:
                        self.log_success(f"Documentation file '{doc_file}' has content")
                    else:
                        self.log_warning(f"Documentation file '{doc_file}' may be empty")
            else:
                self.log_error(f"Documentation file '{doc_file}' not found")
        
        return len(self.errors) == 0

    def run_comprehensive_validation(self) -> bool:
        """Run all validation checks"""
        logger.info("Starting comprehensive project validation...")
        
        # Run all validations
        structure_valid = self.validate_project_structure()
        docker_valid = self.validate_docker_configurations()
        k8s_valid = self.validate_kubernetes_configurations()
        backend_valid = self.validate_backend_structure()
        frontend_valid = self.validate_frontend_structure()
        docs_valid = self.validate_documentation()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE PROJECT VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Successful checks: {self.success_count}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info(f"Errors: {len(self.errors)}")
        
        if self.warnings:
            logger.info("\nWARNINGS:")
            for warning in self.warnings:
                logger.info(f"  ⚠ {warning}")
                
        if self.errors:
            logger.info("\nERRORS:")
            for error in self.errors:
                logger.info(f"  ✗ {error}")
        else:
            logger.info("\n✓ All project validations passed!")
            
        return len(self.errors) == 0

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    validator = ProjectValidator(project_root)
    
    success = validator.run_comprehensive_validation()
    
    if success:
        logger.info("✓ Project validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Project validation found issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()