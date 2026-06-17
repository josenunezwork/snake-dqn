# Test Suite

This directory contains tests for the Snake Apex project.

## Running Tests

### Option 1: With pytest (Recommended)
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_policy.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Option 2: Standalone (No pytest required)
```bash
# Run simple policy system tests
python test_policy_simple.py
```

## Test Files

### Core Game Tests
- `test_snake.py` - Base Snake class tests
- `test_game_logic.py` - Game logic and collision tests
- `test_game_config.py` - Configuration tests
- `test_human_snake.py` - Human-controlled snake tests

### Model Tests
- `test_apex_network.py` - Apex neural network architecture tests

### Data Tests
- `test_memory_db_handler.py` - Database handler tests

### Policy System Tests
- `test_policy.py` - Policy factory and Apex policy tests
- `test_ai_snake.py` - AISnake integration with Apex policy

## What's Tested

### Apex Policy System
- ApexPolicy creation via PolicyFactory
- ApexPolicy implements all interface methods
- AISnake integrates with Apex Policy interface
- GameState assigns Apex policy to snakes
- Checkpoint save/load includes policy metadata
- Backward compatibility with old checkpoints
- Policy loading from saved models

### Core Functionality
- Snake movement and collisions
- Game logic and food consumption
- Neural network forward/backward passes
- Database operations
- Human snake controls

## Adding New Tests

When extending Apex functionality, add tests in `test_policy.py`:

```python
def test_apex_policy_creation():
    """Test Apex policy creation."""
    policy = PolicyFactory.create_policy('apex', 58, 128, 6)
    assert policy.get_policy_name() == 'apex'

def test_apex_checkpoint_save_load():
    """Test Apex checkpoint serialization."""
    policy = PolicyFactory.create_policy('apex', 58, 128, 6)
    state_dict = policy.get_state_dict()

    new_policy = PolicyFactory.create_policy('apex', 58, 128, 6)
    new_policy.load_state_dict(state_dict)

    assert new_policy.get_policy_name() == 'apex'
```

## Continuous Integration

If using CI/CD, run:
```bash
pytest tests/ --cov=src --cov-report=xml
```

This generates coverage reports compatible with most CI systems.
