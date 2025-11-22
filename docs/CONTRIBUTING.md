# Contributing Guide

Welcome to the Financial Intelligence System project! This guide will help you understand how to contribute effectively to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Standards](#code-standards)
4. [Development Workflow](#development-workflow)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)
9. [Security](#security)
10. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Git**: Version control system
- **Docker**: For containerized development
- **Python 3.8+**: Backend development
- **Node.js 16+**: Frontend development
- **PostgreSQL**: Database (or use Docker)
- **Redis**: Cache and message broker (or use Docker)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/financial-intelligence-system.git
   cd financial-intelligence-system
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-repo/financial-intelligence-system.git
   ```

## Development Environment

### Quick Setup with Docker

The fastest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd financial-intelligence-system

# Copy environment file
cp .env.example .env

# Start development environment
docker compose up --build

# Access the application
# Frontend: http://localhost:5173
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Manual Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python setup_database.py

# Start development server
uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Development Tools

#### Recommended IDE Extensions

**VS Code Extensions:**
- Python
- Pylance
- Black Formatter
- ES7+ React/Redux/React-Native snippets
- Tailwind CSS IntelliSense
- Docker
- GitLens

#### Code Formatting

```bash
# Backend formatting
cd backend
black app/
isort app/
flake8 app/

# Frontend formatting
cd frontend
npm run format
npm run lint
```

## Code Standards

### Python Code Standards

#### Style Guide
- Follow **PEP 8** style guide
- Use **Black** for code formatting
- Use **isort** for import sorting
- Maximum line length: **88 characters**

#### Type Hints
```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

def process_document(
    document_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a financial document with optional parameters."""
    pass
```

#### Docstrings
Use Google-style docstrings:

```python
def analyze_sentiment(text: str, model: str = "finbert") -> Dict[str, float]:
    """Analyze sentiment of financial text.
    
    Args:
        text: The text to analyze
        model: The model to use for analysis
        
    Returns:
        Dictionary containing sentiment scores
        
    Raises:
        ValueError: If text is empty or model is invalid
    """
    pass
```

#### Error Handling
```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def process_request(data: dict) -> dict:
    try:
        # Process the request
        result = await some_operation(data)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### JavaScript/React Code Standards

#### Style Guide
- Use **Prettier** for code formatting
- Use **ESLint** for linting
- Prefer **functional components** with hooks
- Use **TypeScript** for type safety (when applicable)

#### Component Structure
```jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';

/**
 * DocumentAnalyzer component for analyzing financial documents
 */
const DocumentAnalyzer = ({ documentId, onAnalysisComplete }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (documentId) {
      analyzeDocument();
    }
  }, [documentId]);

  const analyzeDocument = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await api.analyzeDocument(documentId);
      setAnalysis(result);
      onAnalysisComplete?.(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) return <div>Analyzing document...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="document-analyzer">
      {/* Component content */}
    </div>
  );
};

DocumentAnalyzer.propTypes = {
  documentId: PropTypes.string.isRequired,
  onAnalysisComplete: PropTypes.func
};

export default DocumentAnalyzer;
```

#### Naming Conventions
- **Components**: PascalCase (`DocumentAnalyzer`)
- **Functions**: camelCase (`analyzeDocument`)
- **Constants**: UPPER_SNAKE_CASE (`API_BASE_URL`)
- **Files**: kebab-case (`document-analyzer.jsx`)

### Database Standards

#### Migration Files
```python
"""Add sentiment analysis table

Revision ID: 001_add_sentiment_table
Revises: 
Create Date: 2024-01-01 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'sentiment_analysis',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('sentiment_score', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'])
    )

def downgrade():
    op.drop_table('sentiment_analysis')
```

#### Query Optimization
- Use **indexes** for frequently queried columns
- Use **EXPLAIN ANALYZE** to optimize queries
- Avoid **N+1 queries** with proper joins
- Use **pagination** for large result sets

## Development Workflow

### Branch Strategy

We use **Git Flow** with the following branches:

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature development branches
- **hotfix/**: Critical bug fixes
- **release/**: Release preparation branches

### Feature Development

1. **Create feature branch**:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/document-sentiment-analysis
   ```

2. **Develop the feature**:
   - Write code following standards
   - Add tests for new functionality
   - Update documentation
   - Commit changes with clear messages

3. **Commit messages**:
   ```bash
   # Format: type(scope): description
   git commit -m "feat(sentiment): add FinBERT sentiment analysis"
   git commit -m "fix(api): handle empty document content"
   git commit -m "docs(readme): update installation instructions"
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/document-sentiment-analysis
   # Create pull request on GitHub
   ```

### Commit Message Convention

Use **Conventional Commits** format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add document upload endpoint
fix(frontend): resolve chart rendering issue
docs(api): update authentication documentation
test(sentiment): add unit tests for sentiment analysis
```

## Testing Guidelines

### Backend Testing

#### Unit Tests
```python
import pytest
from app.services.sentiment_service import SentimentAnalyzer

class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        text = "The company reported excellent quarterly results"
        result = self.analyzer.analyze(text)
        
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.8
    
    def test_negative_sentiment(self):
        text = "The company faces significant challenges"
        result = self.analyzer.analyze(text)
        
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.7
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        texts = ["Good news", "Bad news", "Neutral news"]
        results = await self.analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all('sentiment' in result for result in results)
```

#### Integration Tests
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestDocumentAPI:
    def test_upload_document(self):
        with open("test_document.pdf", "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": f},
                data={
                    "company": "Test Corp",
                    "document_type": "10-K",
                    "filing_date": "2024-01-01"
                }
            )
        
        assert response.status_code == 200
        assert "document_id" in response.json()
    
    def test_search_documents(self):
        response = client.post(
            "/api/documents/search",
            json={
                "query": "revenue growth",
                "companies": ["Test Corp"]
            }
        )
        
        assert response.status_code == 200
        assert "results" in response.json()
```

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_sentiment.py

# Run tests with specific marker
pytest -m "not slow"
```

### Frontend Testing

#### Component Tests
```jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DocumentAnalyzer from '../DocumentAnalyzer';

describe('DocumentAnalyzer', () => {
  test('renders loading state', () => {
    render(<DocumentAnalyzer documentId="123" />);
    expect(screen.getByText('Analyzing document...')).toBeInTheDocument();
  });

  test('handles analysis completion', async () => {
    const mockOnComplete = jest.fn();
    render(
      <DocumentAnalyzer 
        documentId="123" 
        onAnalysisComplete={mockOnComplete} 
      />
    );

    await waitFor(() => {
      expect(mockOnComplete).toHaveBeenCalledWith(
        expect.objectContaining({
          sentiment: expect.any(String),
          confidence: expect.any(Number)
        })
      );
    });
  });

  test('displays error message', async () => {
    // Mock API error
    jest.spyOn(api, 'analyzeDocument').mockRejectedValue(
      new Error('Analysis failed')
    );

    render(<DocumentAnalyzer documentId="123" />);

    await waitFor(() => {
      expect(screen.getByText('Error: Analysis failed')).toBeInTheDocument();
    });
  });
});
```

#### Running Frontend Tests
```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test DocumentAnalyzer.test.jsx
```

### Test Coverage

Maintain **minimum 80% test coverage** for:
- Critical business logic
- API endpoints
- Database operations
- Frontend components

## Documentation

### Code Documentation

#### API Documentation
- Use **OpenAPI/Swagger** for API documentation
- Document all endpoints with examples
- Include error responses and status codes
- Provide request/response schemas

#### Inline Documentation
```python
class DocumentProcessor:
    """Processes financial documents for analysis.
    
    This class handles document upload, text extraction,
    and preparation for various analysis tasks.
    """
    
    def __init__(self, config: ProcessorConfig):
        """Initialize the document processor.
        
        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
    
    async def process_document(
        self, 
        file_path: str, 
        metadata: DocumentMetadata
    ) -> ProcessingResult:
        """Process a document and extract relevant information.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata including company and type
            
        Returns:
            ProcessingResult containing extracted text and metadata
            
        Raises:
            ProcessingError: If document processing fails
            ValidationError: If metadata is invalid
        """
        pass
```

### README Updates

When adding new features, update relevant documentation:

- **Installation instructions**
- **Configuration options**
- **Usage examples**
- **API changes**
- **Breaking changes**

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run tests**:
   ```bash
   # Backend tests
   cd backend && pytest

   # Frontend tests
   cd frontend && npm test

   # Integration tests
   docker compose -f docker-compose.test.yml up --abort-on-container-exit
   ```

3. **Check code quality**:
   ```bash
   # Backend
   black app/ && isort app/ && flake8 app/

   # Frontend
   npm run lint && npm run format
   ```

### PR Template

Use this template for pull requests:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] No breaking changes (or breaking changes documented)

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated checks** must pass
2. **At least one reviewer** approval required
3. **All conversations** must be resolved
4. **Tests must pass** in CI/CD pipeline
5. **Documentation** must be updated

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Windows 10, macOS 12.0]
- Browser: [e.g. Chrome 96, Firefox 95]
- Version: [e.g. 1.0.0]

**Additional Context**
Screenshots, logs, or other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How would you like this feature to work?

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

## Security

### Reporting Security Issues

**DO NOT** create public issues for security vulnerabilities.

Instead:
1. Email security concerns to: `security@your-domain.com`
2. Include detailed description of the vulnerability
3. Provide steps to reproduce (if applicable)
4. Allow time for investigation and fix before disclosure

### Security Best Practices

- **Never commit secrets** (API keys, passwords, tokens)
- **Use environment variables** for configuration
- **Validate all inputs** on both client and server
- **Use HTTPS** for all communications
- **Keep dependencies updated** regularly
- **Follow OWASP guidelines** for web security

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** and considerate
- **Be collaborative** and helpful
- **Be patient** with newcomers
- **Focus on constructive feedback**
- **Respect different viewpoints**

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code review and collaboration
- **Email**: Security issues and private matters

### Getting Help

If you need help:

1. **Check documentation** first
2. **Search existing issues** for similar problems
3. **Ask in GitHub Discussions** for general questions
4. **Create an issue** for bugs or feature requests
5. **Join community discussions** for broader topics

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Project documentation** for major features
- **Community highlights** for exceptional contributions

Thank you for contributing to the Financial Intelligence System! Your contributions help make this project better for everyone.