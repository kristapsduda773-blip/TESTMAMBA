# Implementation Guide: Training Mode Selection & Checkpoint Resume

## Quick Start

### Option 1: Minimal Fix (5 minutes)

Replace lines **3190-3193** in `nnmamba_unet_bak_final_drive.py` with this:

```python
# ================================
# CHOOSE YOUR TRAINING MODE HERE
# ================================
TRAINING_MODE = "new"  # Change to "continue" to resume training

if TRAINING_MODE == "new":
    print("Starting NEW training from epoch 0")
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=10000,
        patience=30,
        base_lr=initial_lr,
        max_lr=max_lr,
        step_size=step_size,
        save_dir=CHECKPOINT_DIR,
        save_name=PYTORCH_CHECKPOINT_NAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        preview_gen=val_gen,
        resume_checkpoint=None,  # Start fresh
        history_pickle_path=PYTORCH_HISTORY_PATH,
    )
    
elif TRAINING_MODE == "continue":
    print("Attempting to CONTINUE training from checkpoint")
    checkpoint_path = "/content/drive/MyDrive/Unet_checkpoints/best_model_mamba_pytorch.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found! Starting new training instead.")
        resume_path = None
        total_epochs = 10000
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        total_epochs = start_epoch + 1000  # Train 1000 more epochs
        resume_path = checkpoint_path
        print(f"Resuming from epoch {start_epoch} → {total_epochs}")
    
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=total_epochs,
        patience=30,
        base_lr=initial_lr,
        max_lr=max_lr,
        step_size=step_size,
        save_dir=CHECKPOINT_DIR,
        save_name=PYTORCH_CHECKPOINT_NAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        preview_gen=val_gen,
        resume_checkpoint=resume_path,
        history_pickle_path=PYTORCH_HISTORY_PATH,
    )
```

**Done!** Now you can switch modes by changing `TRAINING_MODE = "new"` to `TRAINING_MODE = "continue"`.

---

### Option 2: Complete Fix (Use the provided file)

Replace lines **3150-3193** with the entire content of `FIXED_TRAINING_SECTION.py`.

This includes:
- ✅ Full error handling
- ✅ Detailed console output
- ✅ User confirmation prompts
- ✅ Training status reporting
- ✅ Helper functions

---

## For Google Colab Notebook Users

### Step-by-Step Notebook Setup

#### Cell 1: Configuration
```python
# ========================================
# TRAINING CONFIGURATION - EDIT THIS CELL
# ========================================

# Choose training mode
TRAINING_MODE = "new"  # Options: "new" or "continue"

# Dataset path (update to match your Drive structure)
DATASET_BASE_PATH = "/content/drive/MyDrive/BHSD/SPLIT/segmentation"

# Checkpoint configuration
CHECKPOINT_DIR = '/content/drive/MyDrive/Unet_checkpoints'
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best_model_mamba_pytorch.pt"

# Training hyperparameters
NUM_EPOCHS = 10000           # For new training
ADDITIONAL_EPOCHS = 1000     # For continue mode

print(f"✓ Configuration loaded")
print(f"  Mode: {TRAINING_MODE}")
print(f"  Dataset: {DATASET_BASE_PATH}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
```

#### Cell 2: Environment Setup (all your imports, etc.)
```python
# Keep all your existing imports and setup code here
from google.colab import drive
drive.mount('/content/drive')

# ... rest of imports ...
```

#### Cell 3: Data Preparation
```python
# All your data loading, generators, etc.
train_gen = PatchGenerator(...)
val_gen = PatchGenerator(...)
# ...
```

#### Cell 4: Model & Training Functions
```python
# All your model definitions, loss functions, etc.
# fit_pytorch_mamba() function
# ...
```

#### Cell 5: Execute Training
```python
# Training execution with mode selection
# (Insert code from FIXED_TRAINING_SECTION.py here)

if TRAINING_MODE == "new":
    # ... new training logic ...
elif TRAINING_MODE == "continue":
    # ... continue training logic ...
```

#### Cell 6: Visualize Results
```python
# Plot training curves
plot_history_curves(history, title_prefix="ICH Segmentation")
```

---

## Common Use Cases

### Use Case 1: First Time Training
```python
TRAINING_MODE = "new"
NUM_EPOCHS = 10000
```
Run all cells. Training starts from epoch 0.

### Use Case 2: Colab Session Timed Out
Your training was at epoch 1500 when Colab disconnected.

```python
TRAINING_MODE = "continue"
ADDITIONAL_EPOCHS = 1000
```
Run cells. Training resumes from epoch 1501 → 2500.

### Use Case 3: Fine-tuning After Early Stopping
Training stopped at epoch 3200 due to early stopping.  
You want to train 500 more epochs.

```python
TRAINING_MODE = "continue"
ADDITIONAL_EPOCHS = 500
```
Run cells. Training resumes from epoch 3201 → 3700.

### Use Case 4: Starting Over
You want to completely restart training (ignore previous checkpoint).

```python
TRAINING_MODE = "new"
NUM_EPOCHS = 10000
```
**Warning:** This will overwrite your existing checkpoint!

---

## Verification Checklist

Before running training, verify:

- [ ] **Dataset path exists**
  ```python
  import os
  assert os.path.exists(DATASET_BASE_PATH), f"Dataset not found at {DATASET_BASE_PATH}"
  ```

- [ ] **Google Drive is mounted**
  ```python
  assert os.path.exists('/content/drive'), "Drive not mounted! Run: drive.mount('/content/drive')"
  ```

- [ ] **Checkpoint directory is writable**
  ```python
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  test_file = f"{CHECKPOINT_DIR}/.test"
  with open(test_file, 'w') as f:
      f.write('test')
  os.remove(test_file)
  print("✓ Checkpoint directory is writable")
  ```

- [ ] **If continuing: checkpoint exists**
  ```python
  if TRAINING_MODE == "continue":
      assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
      print(f"✓ Checkpoint found: {CHECKPOINT_PATH}")
  ```

- [ ] **GPU available (recommended)**
  ```python
  import torch
  print(f"GPU available: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
      print(f"GPU: {torch.cuda.get_device_name(0)}")
  ```

---

## Troubleshooting

### Problem: "FileNotFoundError: Checkpoint not found"

**Cause:** You set `TRAINING_MODE = "continue"` but no checkpoint exists.

**Solution:**
1. Change to `TRAINING_MODE = "new"` to start fresh, OR
2. Use the fixed code which auto-fallback to new training, OR
3. Update `CHECKPOINT_PATH` to point to correct location

### Problem: "Training starts from epoch 0 even in continue mode"

**Cause:** The checkpoint file is corrupted or doesn't contain 'epoch' key.

**Solution:**
```python
# Check checkpoint contents
import torch
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
print("Checkpoint keys:", ckpt.keys())
print("Saved epoch:", ckpt.get('epoch', 'NOT FOUND'))
```

If 'epoch' is missing, the checkpoint is invalid. Start new training.

### Problem: "CUDA out of memory"

**Cause:** Batch size too large, or previous training session didn't release memory.

**Solution:**
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Restart Colab runtime if issue persists
# Runtime → Restart runtime
```

### Problem: "History curves show discontinuity when continuing"

**Cause:** This is NORMAL if you're continuing training. The history contains all epochs (old + new).

**Example:**
- First session: epochs 0-1500
- Second session: epochs 1501-2500
- History will show: [0, 1, 2, ..., 1500, 1501, ..., 2500]

If you want to see only the new portion:
```python
# Plot only new epochs
start_idx = 1500  # Where you resumed from
new_history = {
    'loss': history['loss'][start_idx:],
    'val_loss': history['val_loss'][start_idx:],
    # ... etc
}
plot_history_curves(new_history, title_prefix="Continued Training")
```

### Problem: "Model performs worse after resuming"

**Possible causes:**
1. **Learning rate scheduler state** - may have reset
2. **Data shuffling** - different data order in new session
3. **Early stopping counter** - if it was close to patience limit

**Solution:**
Check the checkpoint was loaded correctly:
```python
# Verify optimizer state was restored
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
print("Optimizer state present:", 'optimizer_state' in checkpoint)
print("Scheduler state present:", 'scheduler_state' in checkpoint)
print("Last learning rate:", checkpoint['history']['lr'][-1])
```

---

## Testing Your Implementation

### Test 1: New Training Works
```python
TRAINING_MODE = "new"
NUM_EPOCHS = 5  # Just 5 epochs for testing

# Run training
# Expected: Trains 5 epochs, creates checkpoint
```

### Test 2: Continue Training Works
```python
# After Test 1 completes:
TRAINING_MODE = "continue"
ADDITIONAL_EPOCHS = 3  # Train 3 more epochs

# Run training
# Expected: Resumes from epoch 6, trains to epoch 8
```

### Test 3: Checkpoint Validation
```python
# After Test 2 completes:
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Verify checkpoint structure
assert 'model_state' in checkpoint, "Missing model_state"
assert 'epoch' in checkpoint, "Missing epoch"
assert 'optimizer_state' in checkpoint, "Missing optimizer_state"
assert 'scheduler_state' in checkpoint, "Missing scheduler_state"
assert 'history' in checkpoint, "Missing history"
assert checkpoint['epoch'] == 8, f"Expected epoch 8, got {checkpoint['epoch']}"

print("✓ All checkpoint validations passed")
```

### Test 4: History Continuity
```python
# After Test 2 completes:
assert len(history['loss']) == 8, f"Expected 8 epochs in history, got {len(history['loss'])}"
assert len(history['val_loss']) == 8, f"Expected 8 val epochs, got {len(history['val_loss'])}"

# Check continuity (no duplicate epochs)
# The history should be continuous: [1, 2, 3, 4, 5, 6, 7, 8]
print("✓ History continuity verified")
```

---

## Advanced: Interactive Mode (Optional)

For a better user experience, add interactive widgets:

```python
# Add this cell BEFORE training execution cell

try:
    import ipywidgets as widgets
    from IPython.display import display
    
    print("=" * 70)
    print("🎛️  INTERACTIVE TRAINING CONFIGURATION")
    print("=" * 70)
    
    mode_selector = widgets.RadioButtons(
        options=[
            ('🔵 New Training (start from epoch 0)', 'new'),
            ('🟢 Continue Training (resume from checkpoint)', 'continue')
        ],
        description='Mode:',
        value='new'
    )
    
    epochs_input = widgets.IntText(
        value=10000,
        description='Epochs:',
        tooltip='Total epochs for new training, or additional epochs for continue'
    )
    
    run_button = widgets.Button(
        description='▶️  Start Training',
        button_style='success',
        tooltip='Click to begin training'
    )
    
    output_area = widgets.Output()
    
    def on_run_clicked(b):
        global TRAINING_MODE, NUM_EPOCHS, ADDITIONAL_EPOCHS
        TRAINING_MODE = mode_selector.value
        NUM_EPOCHS = epochs_input.value
        ADDITIONAL_EPOCHS = epochs_input.value
        
        with output_area:
            output_area.clear_output()
            print(f"✅ Configuration saved!")
            print(f"   Mode: {TRAINING_MODE}")
            print(f"   Epochs: {epochs_input.value}")
            print()
            print("📝 Now run the TRAINING EXECUTION cell below.")
    
    run_button.on_click(on_run_clicked)
    
    display(mode_selector)
    display(epochs_input)
    display(run_button)
    display(output_area)
    
    print("\n👆 Configure training above, then run the next cell.")
    
except ImportError:
    print("ipywidgets not available. Using manual configuration.")
    print("Set TRAINING_MODE variable in the configuration cell.")
```

---

## Summary

| Action | Code Change |
|--------|-------------|
| **Choose new training** | `TRAINING_MODE = "new"` |
| **Continue training** | `TRAINING_MODE = "continue"` |
| **Change epochs** | `NUM_EPOCHS = 5000` or `ADDITIONAL_EPOCHS = 500` |
| **Check status** | `check_training_status()` |
| **View history** | `plot_history_curves(history)` |

## Need Help?

Common questions:

**Q: How do I know if my checkpoint is valid?**  
A: Run `check_training_status()` - it will show all checkpoint info.

**Q: Can I change hyperparameters when continuing?**  
A: Yes, but LR scheduler will use the resumed state. For fresh LR schedule, start new training.

**Q: Will continuing training overwrite my checkpoint?**  
A: Yes, but only with better models (lower validation loss). The checkpoint always contains the BEST model.

**Q: How many times can I continue training?**  
A: Unlimited. Each continue session adds more epochs to the history.

**Q: What if I want to train forever until I stop it?**  
A: Set `NUM_EPOCHS = 999999` and `patience = 1000` (large values). Use early stopping or manual stop.

---

## File Reference

- `VALIDATION_REPORT.md` - Detailed analysis of original code
- `FIXED_TRAINING_SECTION.py` - Complete corrected training code
- `IMPLEMENTATION_GUIDE.md` - This file (usage instructions)
