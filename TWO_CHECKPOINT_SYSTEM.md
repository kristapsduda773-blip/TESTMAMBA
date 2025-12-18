# Two Checkpoint System Implemented

## Overview

The training now saves TWO separate checkpoint files:

1. **`checkpoint_latest.pt`** - Saved after EVERY epoch
   - For resuming training
   - Contains: model, optimizer, scheduler, history, epoch counter
   
2. **`checkpoint_best.pt`** - Saved only when validation IMPROVES
   - For inference/evaluation
   - Contains: best model weights, epoch number, validation loss

## File Structure

```
/content/drive/MyDrive/Unet_checkpoints/
├── checkpoint_latest.pt    # Resume training from here
├── checkpoint_best.pt      # Use this for inference
└── pytorch_training_history.pkl
```

## How It Works

### During Training

```
Epoch 1: train -> val_loss=0.150 (NEW BEST!)
  - Save checkpoint_best.pt (epoch 1, val_loss=0.150)
  - Save checkpoint_latest.pt (epoch 1, all states)

Epoch 2: train -> val_loss=0.145 (NEW BEST!)
  - Save checkpoint_best.pt (epoch 2, val_loss=0.145)
  - Save checkpoint_latest.pt (epoch 2, all states)

Epoch 3: train -> val_loss=0.148 (worse, no improvement)
  - checkpoint_best.pt unchanged (still epoch 2)
  - Save checkpoint_latest.pt (epoch 3, all states)

Epoch 4: train -> val_loss=0.152 (worse)
  - checkpoint_best.pt unchanged (still epoch 2)
  - Save checkpoint_latest.pt (epoch 4, all states)

[Colab timeout - training interrupted]
```

### When Resuming

```
Look for: checkpoint_latest.pt
  - Exists: Load epoch 4 model
  - Resume from epoch 5

Also look for: checkpoint_best.pt
  - Track that best model is from epoch 2
  
Continue training from epoch 5, 6, 7, ...
```

### Console Output

```
[Epoch 98/10000] Finished training data pass.
[Epoch 98/10000] Finished validation data pass.
Epoch 98/10000 COMPLETE - loss=0.1234 val_loss=0.1567
  Latest checkpoint saved (epoch 98) -> checkpoint_latest.pt
  Epochs without improvement: 5/30

[Epoch 99/10000] Finished training data pass.
[Epoch 99/10000] Finished validation data pass.
Epoch 99/10000 COMPLETE - loss=0.1201 val_loss=0.1523
  NEW BEST MODEL - val_loss improved to 0.1523
  Saved to: checkpoint_best.pt
  Latest checkpoint saved (epoch 99) -> checkpoint_latest.pt
  Epochs without improvement: 0/30
```

## Checkpoint Contents

### checkpoint_latest.pt
```python
{
    'model_state': <model_from_epoch_100>,
    'epoch': 100,
    'history': {'loss': [...], 'val_loss': [...], ...},
    'best_val': 0.1234,  # Best val_loss seen so far
    'no_improve': 5,     # Early stopping counter
    'optimizer_state': {...},
    'scheduler_state': {...},
    'scaler_state': {...}  # If CUDA
}
```

### checkpoint_best.pt
```python
{
    'model_state': <model_from_epoch_80>,  # When val_loss was lowest
    'epoch': 80,
    'history': {'loss': [...], 'val_loss': [...], ...},
    'best_val': 0.1234
}
```

## Usage

### Resume Training
```python
TRAINING_MODE = "continue"
# Automatically loads checkpoint_latest.pt
# Resumes from last completed epoch
```

### Load Best Model for Inference
```python
import torch

# Load the best model
checkpoint = torch.load('/content/drive/MyDrive/Unet_checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Use for predictions
with torch.no_grad():
    predictions = model(test_data)
```

### Check Which Epoch is Best
```python
import torch

best_ckpt = torch.load('checkpoint_best.pt', map_location='cpu')
latest_ckpt = torch.load('checkpoint_latest.pt', map_location='cpu')

print(f"Latest epoch: {latest_ckpt['epoch']}")
print(f"Best epoch: {best_ckpt['epoch']}")
print(f"Best val_loss: {best_ckpt['best_val']:.4f}")
```

## Benefits

1. **Never lose progress** - Latest checkpoint saved every epoch
2. **Best model preserved** - Separate file with lowest val_loss
3. **Flexible resume** - Always continue from where you left off
4. **Easy inference** - Just load checkpoint_best.pt
5. **Clear separation** - Training state vs. best model

## Storage Impact

**Two files instead of one:**
- `checkpoint_latest.pt`: ~500MB (model + optimizer + scheduler)
- `checkpoint_best.pt`: ~300MB (model only, no optimizer)
- Total: ~800MB per training run

This is acceptable given the benefits.

## Resume Example

```
MODE: CONTINUE TRAINING
----------------------------------------------------------------------
Loading LATEST checkpoint from: checkpoint_latest.pt
Checkpoint loaded successfully!

CHECKPOINT INFORMATION:
   - Epochs completed so far: 100
   - Next epoch to train: 101
   - Will train until epoch: 1100
   - Additional epochs: 1000

Loaded checkpoint: 100 epochs completed. Will resume from epoch 101.
Resuming from LATEST checkpoint (epoch 100)
Best model saved separately with val_loss=0.1234

Starting training loop: epochs 101 to 1100

[Epoch 101/1100] Finished training data pass.
...
```

## After Training Completes

```
Training complete. Loading best model weights (lowest val_loss)...
Best model was from an earlier epoch with val_loss=0.1234

CONTINUED TRAINING COMPLETED
   Resumed from epoch: 100
   Total epochs in history: 1100
   Latest checkpoint: checkpoint_latest.pt
   Best checkpoint: checkpoint_best.pt
======================================================================

Training session finished!
   Mode: CONTINUE
   Device: cuda
   Latest checkpoint exists: True
   Best checkpoint exists: True

To resume training, use the LATEST checkpoint.
To run inference/testing, use the BEST checkpoint.
```

## Summary

Two-file system gives you:
- **checkpoint_latest.pt** - Resume training from last epoch
- **checkpoint_best.pt** - Use best model for evaluation

You always resume from LATEST, never from BEST.
This prevents losing training progress while preserving the best model.
