# Quick Reference Card

## Your File Has Been Updated

`nnmamba_unet_bak_final_drive.py` now has user-selectable training modes.

---

## How to Control Training Mode

### Edit Line 440 in the .py file:

```python
TRAINING_MODE = "new"       # Start fresh from epoch 0
# or
TRAINING_MODE = "continue"  # Resume from last checkpoint
```

**That's it!** Just change this one variable.

---

## Quick Actions

### Start New Training
1. Line 440: `TRAINING_MODE = "new"`
2. Run script
3. Trains from epoch 0

### Continue Training (After Colab Timeout)
1. Line 440: `TRAINING_MODE = "continue"`
2. Run script  
3. Resumes from last epoch

### Change Number of Epochs
- **New training:** Line 3171: `NUM_EPOCHS = 10000`
- **Continue training:** Line 3172: `ADDITIONAL_EPOCHS = 1000`

### Check Checkpoint Status
Add this cell in your notebook:
```python
check_training_status()
```
Shows epoch, loss, and all saved states.

---

## Key Line Numbers

| What | Line |
|------|------|
| **Training mode** | 440 |
| **Total epochs (new)** | 3171 |
| **Additional epochs (continue)** | 3172 |

---

## Common Scenarios

### Scenario 1: First Time Training
```python
TRAINING_MODE = "new"
NUM_EPOCHS = 10000
```
→ Trains for 10,000 epochs (or until early stopping)

### Scenario 2: Colab Timed Out at Epoch 2000
```python
TRAINING_MODE = "continue"
ADDITIONAL_EPOCHS = 1000
```
→ Resumes from epoch 2001, trains to epoch 3000

### Scenario 3: Want to Train More After Finishing
```python
TRAINING_MODE = "continue"
ADDITIONAL_EPOCHS = 500
```
→ Adds 500 more epochs to previous training

### Scenario 4: Start Over (Ignore Old Checkpoint)
```python
TRAINING_MODE = "new"
```
→ Starts fresh, overwrites old checkpoint

---

## Safety Features

- **Checkpoint missing:** Automatically falls back to new training (no crash)
- **Existing checkpoint:** Shows warning before overwriting
- **Invalid mode:** Clear error message with instructions
- **Detailed logging:** Know exactly what's happening  

---

## Test Before Full Training

Always test with small epochs first:

```python
TRAINING_MODE = "new"
NUM_EPOCHS = 5  # Just 5 epochs for testing
```

Run script, verify it works, then change to full epochs.

---

## Console Output Examples

### New Training Mode:
```
======================================================================
NEURAL NETWORK TRAINING - ICH SEGMENTATION
======================================================================

MODE: NEW TRAINING
----------------------------------------------------------------------
• Starting from epoch 0 with fresh model weights
• Optimizer and scheduler initialized to default states
• Training will run for 10000 epochs (or until early stopping)
• Checkpoints will be saved to: /content/drive/MyDrive/Unet_checkpoints
----------------------------------------------------------------------
```

### Continue Training Mode:
```
======================================================================
NEURAL NETWORK TRAINING - ICH SEGMENTATION
======================================================================

MODE: CONTINUE TRAINING
----------------------------------------------------------------------
📂 Loading checkpoint from: /content/drive/.../checkpoint.pt
✅ Checkpoint loaded successfully!

📊 CHECKPOINT INFORMATION:
   • Last completed epoch: 2000
   • Training will resume from: epoch 2001
   • Target epoch: 3000
   • Additional epochs to train: 1000
   • Best validation loss so far: 0.0234

🔄 RESTORING MODEL STATE:
   ✓ Model weights
   ✓ Optimizer state (SGD with momentum)
   ✓ Learning rate scheduler (Cyclic LR)
   ✓ Training history
   ✓ Early stopping counter
   ✓ AMP scaler state
----------------------------------------------------------------------
```

---

## Emergency Fixes

### Problem: Code crashes immediately
**Fix:** Check line 440 - should be `TRAINING_MODE = "new"` or `"continue"` (lowercase)

### Problem: Always trains from epoch 0
**Fix:** Change line 440 to `TRAINING_MODE = "continue"`

### Problem: Can't find checkpoint
**Fix:** Check path in console output, update `CHECKPOINT_DIR` if needed

---

## Documentation Reference

| Issue | Check File |
|-------|------------|
| How to use | `CHANGES_APPLIED.md` |
| Detailed guide | `IMPLEMENTATION_GUIDE.md` |
| Technical details | `VALIDATION_REPORT.md` |
| Overview | `SUMMARY.md` |

---

## What Changed in Your File

**Added (line 440):**
```python
TRAINING_MODE = "new"  # Your control variable
```

**Replaced (lines 3163-3361):**
- Old: Hardcoded `continue_training_block()` function
- New: Smart conditional logic based on `TRAINING_MODE`

**Added (lines 3373-3455):**
```python
check_training_status()  # New helper function
```

---

## Pro Tips

1. **Test First:** Always run 5 epochs test before full training
2. **Check Status:** Use `check_training_status()` to verify checkpoint
3. **Monitor GPU:** Ensure GPU is available before long runs
4. **Save Often:** Default settings save on every improvement
5. **Resume Anytime:** Continue mode handles all the complexity

---

## Ready to Train

Your code is now production-ready with proper checkpoint management.

**To start:**
1. Open `nnmamba_unet_bak_final_drive.py`
2. Set line 440: `TRAINING_MODE = "new"` or `"continue"`
3. Run in Google Colab
4. Watch your model train

---

**Need help?** Read `CHANGES_APPLIED.md` for full details.

**Questions about original issues?** Read `VALIDATION_REPORT.md`.

**Step-by-step guide?** Read `IMPLEMENTATION_GUIDE.md`.
