#!/usr/bin/env zsh
# Quick Training Script
# Generates experiences and trains offline in one command

set -e  # Exit on error

# Auto-activate venv if it exists and not already activated
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

echo "🚀 Quick Training Pipeline"
echo "=========================="
echo ""

# Default parameters
EPISODES=${1:-1000}

echo "📝 Configuration:"
echo "   Episodes: $EPISODES"
echo ""

echo "⚡ Training in headless mode..."
python src/main.py --headless --episodes $EPISODES

echo ""
echo "✅ Training complete!"
echo ""
echo "📊 Check training_metrics.png for results"
echo ""
echo "🎮 Now run: python src/main.py"
echo "   (The UI will auto-load the best model)"