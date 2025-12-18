# Epoch Numbering Clarification

## How Epoch Counting Works

### Understanding the Numbers

There are three ways to think about epochs:

1. **Epoch INDEX** (0-based, internal): 0, 1, 2, 3, ...
2. **Epoch DISPLAY** (1-based, user-facing): 1, 2, 3, 4, ...
3. **Epochs COMPLETED** (count): 1, 2, 3, 4, ...

### What Gets Saved in Checkpoint

The checkpoint saves **"epochs completed"**:
```python
'epoch': epoch + 1  # If epoch index=0, saves 1 (meaning "1 epoch done")
```

### Example: Training 1 Epoch

**During training:**
- Loop iteration: `epoch = 0` (index)
- Console shows: `Epoch 1/10000` (display)
- After completion, checkpoint saves: `'epoch': 1` (count)

**When resuming:**
- Loads checkpoint: `start_epoch = 1` (1 epoch was completed)
- Console message: "Epochs completed so far: 1"
- Console message: "Next epoch to train: 2"
- Loop resumes: `epoch = 1` (index)
- Console shows: `Epoch 2/10000` (display)

### Your Specific Case

If you completed **Epoch 1** (display), then:

1. **What was saved:**
   - Checkpoint field: `'epoch': 1`
   - Meaning: "1 epoch completed"

2. **When you continue:**
   - Loads: `start_epoch = 1`
   - Message: "Epochs completed so far: 1"
   - Message: "Next epoch to train: 2"
   - Training loop starts at index 1, displays as "Epoch 2/10000"

**This is correct behavior.** After completing Epoch 1, you resume from Epoch 2.

## Console Output Examples

### New Training
```
Starting training loop: epochs 1 to 10000

[Epoch 1/10000] Finished training data pass.
[Epoch 1/10000] Finished validation data pass.
Epoch 1/10000 COMPLETE - loss=0.1234 val_loss=0.1456 ...
  Checkpoint saved (best model after epoch 1)
  Epochs without improvement: 0/30

[Epoch 2/10000] Finished training data pass.
[Epoch 2/10000] Finished validation data pass.
Epoch 2/10000 COMPLETE - loss=0.1123 val_loss=0.1345 ...
```

### Continue Training (After Epoch 1 Complete)
```
CHECKPOINT INFORMATION:
   - Epochs completed so far: 1
   - Next epoch to train: 2
   - Will train until epoch: 1001
   
Loaded checkpoint: 1 epochs completed. Will resume from epoch 2.
Starting training loop: epochs 2 to 1001

[Epoch 2/1001] Finished training data pass.
[Epoch 2/1001] Finished validation data pass.
Epoch 2/1001 COMPLETE - loss=0.1098 val_loss=0.1298 ...
```

## Verification

To verify correct behavior:

```python
import torch
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"Epochs in checkpoint: {ckpt['epoch']}")
print(f"Should resume from: {ckpt['epoch'] + 1}")
```

## Summary

The code is working correctly:
- Completes Epoch 1 → Saves 'epoch': 1
- Resumes → Loads 'epoch': 1 → Starts from Epoch 2

The updated console messages now make this crystal clear.
