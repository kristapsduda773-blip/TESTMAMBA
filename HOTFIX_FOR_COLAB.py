"""
HOTFIX: Run this cell in Colab to fix the NameError
========================================================

The error occurs because you're running old code that's still in memory.
"""

# Quick fix - redefine the training function with correct indentation
import torch
import copy
import os

def fit_pytorch_mamba_fixed(
    train_gen,
    val_gen,
    num_epochs=10000,
    patience=30,
    base_lr=1e-5,
    max_lr=1e-2,
    step_size=670,
    save_dir='/content/drive/MyDrive/Unet_checkpoints',
    save_name_latest="checkpoint_latest.pt",
    save_name_best="checkpoint_best.pt",
    class_weights_list=(1.0,),
    device_str=None,
    preview_gen=None,
    resume_checkpoint=None,
    history_pickle_path=None,
):
    """Fixed version with correct indentation"""
    # ... (rest of function - this is just a stub)
    # The actual fix is to restart kernel and reload the file
    pass

print("""
======================================================================
HOTFIX INSTRUCTIONS
======================================================================

The NameError occurs because Colab is running old code from memory.

SOLUTION 1 (Recommended - Quick):
----------------------------------
1. In Colab menu: Runtime → Restart runtime
2. Re-run ALL cells from the beginning
3. Training will work correctly

SOLUTION 2 (Alternative - Manual):
-----------------------------------
1. Find the cell that defines fit_pytorch_mamba() function
2. Delete that cell
3. Copy the updated function from nnmamba_unet_bak_final_drive.py
4. Paste into new cell and run it
5. Then re-run training cell

SOLUTION 3 (If using .py file directly):
-----------------------------------------
1. Make sure you've uploaded the updated nnmamba_unet_bak_final_drive.py
2. Restart runtime
3. Re-import/run the file

The error is NOT in the saved file - it's correct.
The error is in Colab's memory because you're running an old version.

======================================================================
""")

# Show the correct code structure for reference
print("The correct structure should be:")
print("""
def fit_pytorch_mamba(...):
    # ... setup code ...
    
    for epoch in range(start_epoch, num_epochs):
        # ... training ...
        
        # Save best checkpoint (when improved)
        if val_loss < best_val:
            best_val = val_loss
            best_checkpoint = {...}
            torch.save(best_checkpoint, ...)
        else:
            no_improve += 1
        
        # Save latest checkpoint EVERY epoch
        latest_checkpoint = {          # ← This should be at same indentation as 'if'
            'model_state': model.state_dict(),
            'epoch': epoch + 1,
            ...
        }
        torch.save(latest_checkpoint, ...)  # ← This line is failing in your old code
""")
