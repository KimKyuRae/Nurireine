# Nurireine Tests

This directory contains the test suite for Nurireine.

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio
```

### Run All Tests
```bash
SKIP_CONFIG_VALIDATION=true pytest
```

### Run Specific Test File
```bash
SKIP_CONFIG_VALIDATION=true pytest tests/unit/test_validation.py
```

### Run with Verbose Output
```bash
SKIP_CONFIG_VALIDATION=true pytest -v
```

### Run with Coverage
```bash
SKIP_CONFIG_VALIDATION=true pytest --cov=Nurireine
```

## Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - Integration tests (future)
- `tests/fixtures/` - Test data and fixtures (future)
- `conftest.py` - Pytest configuration and shared fixtures

## Current Test Coverage

- ✅ validation.py - Input and config validation
- ⏳ memory.py - Memory management (planned)
- ⏳ gatekeeper.py - SLM analysis (planned)
- ⏳ utils/ - Utility functions (planned)

## Notes

- The `SKIP_CONFIG_VALIDATION` environment variable must be set to skip API key validation during tests
- Tests are designed to be run without Discord.py or AI model dependencies
- See `TESTING.md` for comprehensive testing guide
