#!/bin/bash
# Fix TensorBoard protobuf compatibility issue

set -e

echo "🔧 Fixing TensorBoard protobuf compatibility..."
echo ""

# Downgrade protobuf to compatible version
echo "Downgrading protobuf to version 3.20.3..."
pip install "protobuf<4" --force-reinstall

echo ""
echo "✅ Fixed! Testing TensorBoard..."
python -c "import tensorboard; print('TensorBoard version:', tensorboard.__version__)"

echo ""
echo "🚀 Starting TensorBoard..."
tensorboard --logdir=./runs --port=6006
