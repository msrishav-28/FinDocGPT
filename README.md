# Financial Intelligence System

A comprehensive AI-powered platform for financial document analysis, market prediction, and investment decision-making.

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd financial-intelligence-system

# Start with Docker Compose (recommended)
docker compose up --build

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

## Documentation

For comprehensive documentation, please see the [docs/](./docs/) directory:

- **[Project Overview](./docs/README.md)** - Complete project documentation
- **[API Reference](./docs/API.md)** - REST API and WebSocket documentation
- **[Architecture Guide](./docs/ARCHITECTURE.md)** - System design and architecture
- **[Deployment Guide](./docs/DEPLOYMENT.md)** - Docker, Kubernetes, and production deployment
- **[Contributing Guide](./docs/CONTRIBUTING.md)** - Development guidelines and standards
- **[CI/CD Guide](./docs/CI_CD.md)** - Continuous integration and deployment

## Technology Stack

- **Backend**: Python, FastAPI, PostgreSQL, Redis, Celery
- **Frontend**: React, Vite, Tailwind CSS, Chart.js
- **ML/AI**: PyTorch, Transformers, Prophet, FinBERT
- **Infrastructure**: Docker, Kubernetes, Nginx, Prometheus

## License

This project is licensed under the MIT License. See the LICENSE file for details.