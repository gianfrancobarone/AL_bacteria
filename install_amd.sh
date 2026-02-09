#!/bin/bash
# Installation script for Bacteria Life Simulation on AMD AI Max 395

echo "=========================================="
echo "Bacteria Life Simulation - AMD AI Max 395"
echo "=========================================="
echo ""
echo "System: AMD AI Max 395 with 128GB unified memory"
echo "Using CPU-optimized PyTorch for training"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

echo ""
echo "Installing dependencies..."
echo ""

# Install CPU-only PyTorch
echo "Installing PyTorch (CPU version)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing numpy, gymnasium, pygame..."
pip install numpy>=1.24.0 gymnasium>=0.29.0 pygame>=2.5.0

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Your AMD AI Max 395 advantages:"
echo "  • 128GB unified memory - scale to large models!"
echo "  • Fast CPU training - ~5-10 min for 1000 episodes"
echo "  • Run multiple experiments in parallel"
echo "  • Ready for 0.25B parameter transformer!"
echo ""
echo "Next steps:"
echo "  1. Test:      python test_system.py"
echo "  2. Visualize: python visualize.py random"
echo "  3. Train:     python train.py"
echo "  4. Watch:     python visualize.py"
echo ""
echo "Happy evolving! 🦠"
