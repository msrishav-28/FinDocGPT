# Requirements Management

This directory contains environment-specific Python requirements files for the Financial Intelligence System.

## Files

- **`base.txt`**: Core dependencies required in all environments
- **`production.txt`**: Production dependencies (includes base.txt + ML/AI packages)
- **`development.txt`**: Development dependencies (includes production.txt + dev tools)

## Usage

### Development Environment
```bash
pip install -r requirements/development.txt
```

### Production Environment
```bash
pip install -r requirements/production.txt
```

### Docker Build
The Dockerfile automatically uses `requirements/production.txt` for production builds.

## Updating Dependencies

1. Add new dependencies to the appropriate file:
   - Core dependencies → `base.txt`
   - ML/AI packages → `production.txt`
   - Development tools → `development.txt`

2. Pin versions for production stability:
   ```
   package>=1.0.0,<2.0.0
   ```

3. Test the updated requirements:
   ```bash
   pip install -r requirements/development.txt
   python -m pytest
   ```

## Dependency Management Best Practices

- Always pin versions in production
- Use version ranges for flexibility in development
- Regularly update dependencies for security patches
- Test all environments after dependency updates
- Document any special installation requirements