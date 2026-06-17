# Snake DQN

A reinforcement learning platform where AI snakes learn to play Snake using Apex DQN. Built with PyTorch and PyQt5.

## Features

- **Apex DQN**: Distributed prioritized experience replay for efficient training
- **Real-time GUI**: Watch AI snakes learn with live training metrics
- **Multi-agent**: Multiple snakes training simultaneously
- **Human play mode**: Play alongside AI snakes with keyboard controls
- **H100/Colab support**: Optimized training scripts for cloud GPUs (30M+ steps/sec)

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, PyTorch 2.0+

## Quick Start

### Watch AI Learn (GUI)
```bash
python src/main.py
```

### Human Play Mode
```bash
python src/main.py --human
```
Use arrow keys to control your snake.

### Headless Training (Fast)
```bash
python src/main.py --headless --episodes 100000
```

## Project Structure

```
src/
├── core/           # Configuration & device management
├── model/          # Neural network architectures
├── training/       # Apex DQN policy & replay buffers
├── game/           # Snake game logic
├── ui/             # PyQt5 GUI
├── scripts/        # CLI training tools
└── main.py         # Entry point

colab/              # H100-optimized Colab training
configs/            # YAML configuration files
saved_snakes/       # Trained model checkpoints
```

## Apex DQN

Apex DQN combines several improvements over vanilla DQN for state-of-the-art performance:

1. **Double DQN** - Reduces value overestimation
2. **Dueling Architecture** - Separate value/advantage streams
3. **Prioritized Experience Replay** - Focus on important transitions
4. **Multi-step Returns** - N-step TD targets for faster learning
5. **Distributional RL** - Learn value distribution (C51)
6. **Noisy Networks** - Parameter noise for exploration

## State Representation

See [CLAUDE.md](CLAUDE.md) for the 58-D state layout.

## Reward System

| Event | Reward |
|-------|--------|
| Eat food | +10 + length bonus |
| Death | -10 |
| Move toward food | +0.1 |
| Move away from food | -0.1 |
| Wall proximity | -0.05 to -2.0 |
| Starvation (100+ frames) | Progressive penalty |

## Configuration

Use YAML files for reproducible experiments:

```bash
python src/main.py --config configs/production.yaml
```

Example config:
```yaml
game:
  width: 1450
  height: 830
  num_snakes: 4

training:
  batch_size: 128
  learning_rate: 0.005
  gamma: 0.99
```

## Advanced Training

### Offline Training
```bash
# Generate replay experiences
python src/scripts/generate_experiences.py --episodes 5000 --parallel

# Train Apex DQN on stored replay data and save saved_snakes/best_apex.pth
python src/scripts/offline_train.py --iterations 20000

# Continue live headless training from the offline checkpoint and generated replay
python src/main.py --headless --episodes 100000 --load best_apex.pth --load-memory-db
```

### Google Colab (H100)
Upload `colab/h100_snake_v2.py` to Colab for high-performance training:
- 256k+ parallel environments
- 30M+ steps/second on H100
- Auto-saves checkpoints to Google Drive

## GUI Features

- **Training Dashboard**: Live loss, reward, epsilon graphs
- **Network Visualizer**: See neural network activations
- **Inspector Panel**: Click snakes to view Q-values and state
- **Vision Cones**: Visualize what each snake "sees"

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Saved Models

Models are saved to `saved_snakes/`:
- `best_snake.pth` - Best overall model

Load a trained model:
```bash
python src/main.py --load saved_snakes/best_snake.pth
```

## Troubleshooting

**Snakes not learning?**
- Check epsilon decay reaches ~0.5 after 10k steps
- Verify rewards are being calculated (check GUI)

**Out of memory?**
- Reduce `MEMORY_SIZE` in config (default: 100,000)
- Lower `BATCH_SIZE` (default: 128)

**Too slow?**
- Use `--headless` mode
- Increase `--num-envs` for parallel training

## References

- [DQN](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
- [Ape-X](https://arxiv.org/abs/1803.00933) - Distributed Prioritized Experience Replay
- [Rainbow](https://arxiv.org/abs/1710.02298) - Combining DQN improvements

## License

MIT
