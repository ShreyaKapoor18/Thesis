# Contributing to This Thesis Repository

Thank you for your interest in contributing! This document provides guidelines for contributing to the research project.

## Code of Conduct

This project is committed to providing a welcoming and inspiring community for all. Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

**Bug Reports:**
- Check if the issue already exists
- Provide a clear description of the bug
- Include steps to reproduce
- Specify your environment (OS, Python version, etc.)
- Attach error messages and logs if available

**Feature Requests:**
- Clearly describe the feature
- Explain the use case
- Provide examples if possible

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow code standards:**
   - Python: [PEP 8](https://www.python.org/dev/peps/pep-0008/)
   - Use meaningful variable names
   - Add docstrings to functions and classes
   - Include comments for complex logic

3. **Testing:**
   - Test your changes thoroughly
   - Ensure no existing tests break
   - Add new tests for new functionality

4. **Commit messages:**
   - Use clear, descriptive commit messages
   - Reference issue numbers if applicable
   - Example: `Fix: Resolve connectivity matrix parsing bug #123`

5. **Push to your fork and submit a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Description:**
   - Clearly describe changes made
   - Reference related issues
   - Include test results
   - Mention any breaking changes

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ShreyaKapoor18/Thesis.git
cd Thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development, also install:
pip install pytest pytest-cov black flake8
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings using Google-style format
- Include examples for complex functions
- Update CONTRIBUTING.md if process changes

## Code Style

### Python Code Style

```python
# Good examples
def calculate_connectivity_matrix(data):
    """Calculate connectivity matrix from preprocessed data.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        
    Returns:
        ndarray: Connectivity matrix of shape (n_features, n_features)
    """
    # Implementation here
    pass

# Use type hints
def process_graph(graph: nx.Graph) -> dict:
    """Process graph and return metrics."""
    pass
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=. tests/

# Code quality check
flake8 src/
black --check src/
```

## Questions?

- Open an issue with the `[question]` tag
- Email: shreya.kapoor@fau.de

---

Thank you for contributing to this research! 🙏
