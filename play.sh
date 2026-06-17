#!/usr/bin/env zsh
# Launch the Snake Game UI
# Auto-activates venv and loads best model

set -e

# Auto-activate venv if it exists
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

echo "🎮 Launching Snake DQN Game..."
echo ""
python src/main.py "$@"

