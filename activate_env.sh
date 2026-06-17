#!/usr/bin/env zsh
# Activate the virtual environment
# Usage: source activate_env.sh

SCRIPT_DIR="${0:a:h}"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo "🐍 Activating Snake DQN virtual environment..."
    source venv/bin/activate
    echo "✅ Environment activated!"
    echo ""
    echo "📦 Python: $(which python)"
    echo "📦 Version: $(python --version)"
    echo ""
    echo "Ready to train! Try:"
    echo "  ./quick_train.sh"
    echo "  python src/main.py"
    echo ""
    echo "To deactivate: deactivate"
else
    echo "❌ venv folder not found!"
    echo "Create it with: python3 -m venv venv"
fi

