#!/bin/bash
# Convenience script to run the game in human control mode

echo "🎮 Starting Snake Game in Human Control Mode"
echo "Use arrow keys (↑ ↓ ← →) to control your snake!"
echo "Your experiences will be saved to the database for AI training."
echo ""

python src/main.py --human

