#!/usr/bin/env python
"""
TensorBoard Writer Service for C++ Integration
Provides proper TensorBoard event file writing with full feature support
"""

import sys
import json
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import signal

class TensorBoardWriterService:
    """
    TensorBoard writer service that listens for commands from C++ and writes proper event files.
    Supports:
    - Scalars (loss, metrics, learning rate)
    - Images (with proper formatting)
    - Histograms (weights, gradients, activations)
    - Graphs (model architecture)
    - Hyperparameters (experiment config)
    - Text logging
    - PR curves
    - Custom scalars layout
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        # Create SummaryWriter with explicit flush_secs for immediate writing
        self.writer = SummaryWriter(str(self.log_dir), flush_secs=1)
        self.running = True
        self.event_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        print(f"[TensorBoard Writer] Initialized at {log_dir}", flush=True)
        print(f"[TensorBoard Writer] Event files will be created in: {self.log_dir}", flush=True)
        print(f"[TensorBoard Writer] Flush interval: 1 second (immediate mode)", flush=True)

        # Create custom scalars layout for better organization
        self.setup_custom_layout()

    def setup_custom_layout(self):
        """Create custom scalar layout for organized visualization"""
        layout = {
            "Training": {
                "Loss": ["Multiline", ["loss/train", "loss/val"]],
                "Loss Components": ["Multiline", [
                    "loss_components/si_loss",
                    "loss_components/grad_loss",
                    "loss_components/smooth_loss",
                    "loss_components/reproj_loss"
                ]],
                "Learning Rate": ["Multiline", ["training/lr"]]
            },
            "Metrics": {
                "Relative Error": ["Multiline", ["metrics/abs_rel", "metrics/sq_rel"]],
                "RMSE": ["Multiline", ["metrics/rmse", "metrics/rmse_log"]],
                "Accuracy": ["Multiline", ["metrics/a1", "metrics/a2", "metrics/a3"]]
            },
            "Model": {
                "Gradients": ["Multiline", ["gradients/norm", "gradients/max", "gradients/min"]],
                "Weights": ["Multiline", ["weights/norm", "weights/sparsity"]]
            }
        }

        self.writer.add_custom_scalars(layout)

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        print("[TensorBoard Writer] Shutting down...", flush=True)
        self.running = False
        self.writer.close()
        sys.exit(0)

    def add_scalar(self, tag: str, value: float, step: int):
        """Add scalar value to TensorBoard"""
        self.writer.add_scalar(tag, value, step)
        # Explicit flush to ensure immediate write
        self.writer.flush()
        self.event_count += 1

        # Log every 100 events to track progress
        if self.event_count % 100 == 0:
            print(f"[TensorBoard Writer] {self.event_count} events written", flush=True)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Add multiple scalars at once"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        self.writer.flush()
        self.event_count += len(tag_scalar_dict)

    def add_image(self, tag: str, img_path: str, step: int):
        """Add image from file path"""
        try:
            # Load image and convert to tensor
            from PIL import Image
            img = Image.open(img_path)
            img_array = np.array(img)

            # Convert to CHW format
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            # Normalize to [0, 1]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor.float() / 255.0

            self.writer.add_image(tag, img_tensor, step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard Writer] Error adding image: {e}", flush=True)

    def add_image_tensor(self, tag: str, img_data: list, step: int, shape: list):
        """Add image from raw tensor data"""
        try:
            img_array = np.array(img_data, dtype=np.float32).reshape(shape)
            img_tensor = torch.from_numpy(img_array)

            # Ensure CHW format
            if len(shape) == 3 and shape[0] not in [1, 3]:
                img_tensor = img_tensor.permute(2, 0, 1)

            self.writer.add_image(tag, img_tensor, step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard Writer] Error adding image tensor: {e}", flush=True)

    def add_histogram(self, tag: str, values: list, step: int):
        """Add histogram from values"""
        try:
            values_array = np.array(values, dtype=np.float32)
            values_tensor = torch.from_numpy(values_array)
            self.writer.add_histogram(tag, values_tensor, step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard Writer] Error adding histogram: {e}", flush=True)

    def add_text(self, tag: str, text: str, step: int):
        """Add text to TensorBoard"""
        self.writer.add_text(tag, text, step)
        self.writer.flush()

    def add_hparams(self, hparam_dict: dict, metric_dict: dict):
        """Add hyperparameters and metrics"""
        self.writer.add_hparams(hparam_dict, metric_dict)
        self.writer.flush()

    def add_pr_curve(self, tag: str, labels: list, predictions: list, step: int):
        """Add precision-recall curve"""
        try:
            labels_tensor = torch.tensor(labels, dtype=torch.bool)
            predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
            self.writer.add_pr_curve(tag, labels_tensor, predictions_tensor, step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard Writer] Error adding PR curve: {e}", flush=True)

    def add_embedding(self, tag: str, mat: list, metadata: list, step: int):
        """Add embeddings for visualization"""
        try:
            mat_tensor = torch.tensor(mat, dtype=torch.float32)
            self.writer.add_embedding(mat_tensor, metadata=metadata, global_step=step, tag=tag)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard Writer] Error adding embedding: {e}", flush=True)

    def process_command(self, command: dict):
        """Process a command from C++"""
        try:
            cmd_type = command.get("type")

            if cmd_type == "scalar":
                self.add_scalar(command["tag"], command["value"], command["step"])

            elif cmd_type == "scalars":
                self.add_scalars(command["main_tag"], command["values"], command["step"])

            elif cmd_type == "image":
                if "path" in command:
                    self.add_image(command["tag"], command["path"], command["step"])
                elif "data" in command:
                    self.add_image_tensor(
                        command["tag"],
                        command["data"],
                        command["step"],
                        command["shape"]
                    )

            elif cmd_type == "histogram":
                self.add_histogram(command["tag"], command["values"], command["step"])

            elif cmd_type == "text":
                self.add_text(command["tag"], command["text"], command["step"])

            elif cmd_type == "hparams":
                self.add_hparams(command["hparams"], command["metrics"])

            elif cmd_type == "pr_curve":
                self.add_pr_curve(
                    command["tag"],
                    command["labels"],
                    command["predictions"],
                    command["step"]
                )

            elif cmd_type == "embedding":
                self.add_embedding(
                    command["tag"],
                    command["mat"],
                    command.get("metadata", None),
                    command["step"]
                )

            elif cmd_type == "shutdown":
                self.shutdown()

            else:
                print(f"[TensorBoard Writer] Unknown command type: {cmd_type}", flush=True)

            return {"status": "success"}

        except Exception as e:
            print(f"[TensorBoard Writer] Error processing command: {e}", flush=True)
            return {"status": "error", "message": str(e)}

    def run(self):
        """Run the service, listening for commands on stdin"""
        print("[TensorBoard Writer] Ready to receive commands", flush=True)
        print("[TensorBoard Writer] Send JSON commands via stdin", flush=True)

        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    # EOF reached
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON command
                command = json.loads(line)
                result = self.process_command(command)

                # Send response
                print(json.dumps(result), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({
                    "status": "error",
                    "message": f"Invalid JSON: {e}"
                }), flush=True)

            except Exception as e:
                print(json.dumps({
                    "status": "error",
                    "message": str(e)
                }), flush=True)

        self.writer.close()
        print("[TensorBoard Writer] Service stopped", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: tensorboard_writer.py <log_dir>", file=sys.stderr)
        sys.exit(1)

    log_dir = sys.argv[1]
    service = TensorBoardWriterService(log_dir)

    try:
        service.run()
    except KeyboardInterrupt:
        service.shutdown()


if __name__ == "__main__":
    main()
