"""
FIXED TRAINING EXECUTION SECTION
=================================

This file contains the corrected training execution logic with user-selectable
training mode. Insert this section into your main script to replace lines 3150-3193.

Key Features:
1. User can choose between "new" or "continue" training
2. Proper error handling if checkpoint doesn't exist
3. Graceful fallback to new training
4. Clear console output showing training status
5. All checkpoint functionality preserved
"""

import torch
import os

# ================================
# TRAINING CONFIGURATION
# ================================
# USER: MODIFY THIS VARIABLE TO CHOOSE TRAINING MODE

TRAINING_MODE = "new"  # Options: "new" or "continue"
# 
# "new"      - Start training from scratch (epoch 0)
#              Ignores any existing checkpoints
#              Creates fresh model weights
#
# "continue" - Resume from last checkpoint
#              Loads model weights, optimizer, scheduler states
#              Continues from last completed epoch
#              Falls back to "new" if checkpoint not found

# Checkpoint configuration
CHECKPOINT_PATH = "/content/drive/MyDrive/Unet_checkpoints/best_model_mamba_pytorch.pt"

# Training hyperparameters
NUM_EPOCHS = 10000           # Total epochs for new training
ADDITIONAL_EPOCHS = 1000     # Additional epochs when continuing (added to saved epoch)

# ================================
# TRAINING MODE VALIDATION
# ================================

if TRAINING_MODE not in ["new", "continue"]:
    raise ValueError(
        f"Invalid TRAINING_MODE: '{TRAINING_MODE}'\n"
        f"Valid options are: 'new' or 'continue'\n"
        f"Please update the TRAINING_MODE variable above."
    )

# ================================
# TRAINING EXECUTION LOGIC
# ================================

print("\n" + "=" * 70)
print("NEURAL NETWORK TRAINING - ICH SEGMENTATION")
print("=" * 70)

if TRAINING_MODE == "new":
    # ============================================================
    # START NEW TRAINING FROM EPOCH 0
    # ============================================================
    
    print("\n🔵 MODE: NEW TRAINING")
    print("-" * 70)
    print("• Starting from epoch 0 with fresh model weights")
    print("• Optimizer and scheduler initialized to default states")
    print("• Any existing checkpoints will be OVERWRITTEN")
    print(f"• Training will run for {NUM_EPOCHS} epochs (or until early stopping)")
    print(f"• Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print("-" * 70 + "\n")
    
    # Check if checkpoint already exists and warn user
    if os.path.exists(CHECKPOINT_PATH):
        print("⚠️  WARNING: Existing checkpoint found!")
        print(f"   Location: {CHECKPOINT_PATH}")
        print("   This checkpoint will be OVERWRITTEN during training.")
        response = input("   Continue with new training? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("   Training cancelled by user.")
            raise SystemExit("User cancelled new training to preserve existing checkpoint.")
        print()
    
    # Start new training
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=NUM_EPOCHS,
        patience=patience,
        base_lr=initial_lr,
        max_lr=max_lr,
        step_size=step_size,
        save_dir=CHECKPOINT_DIR,
        save_name=PYTORCH_CHECKPOINT_NAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        preview_gen=val_gen,
        resume_checkpoint=None,  # Critical: None = start from scratch
        history_pickle_path=PYTORCH_HISTORY_PATH,
    )
    
    print("\n" + "=" * 70)
    print("✅ NEW TRAINING COMPLETED")
    print(f"   Total epochs trained: {len(history['loss'])}")
    print(f"   Final training loss: {history['loss'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best validation loss: {min(history['val_loss']):.4f}")
    print(f"   Model saved to: {CHECKPOINT_PATH}")
    print("=" * 70 + "\n")

elif TRAINING_MODE == "continue":
    # ============================================================
    # CONTINUE TRAINING FROM CHECKPOINT
    # ============================================================
    
    print("\n🟢 MODE: CONTINUE TRAINING")
    print("-" * 70)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print("⚠️  CHECKPOINT NOT FOUND!")
        print(f"   Expected location: {CHECKPOINT_PATH}")
        print()
        print("📌 FALLBACK: Starting NEW TRAINING from epoch 0")
        print(f"   Training will run for {NUM_EPOCHS} epochs")
        print("-" * 70 + "\n")
        
        # Fallback to new training
        resume_path = None
        total_epochs = NUM_EPOCHS
        start_epoch = 0
        
    else:
        # Load checkpoint to inspect saved state
        try:
            print(f"📂 Loading checkpoint from: {CHECKPOINT_PATH}")
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
            
            # Extract checkpoint information
            start_epoch = ckpt.get("epoch", 0)
            saved_history = ckpt.get("history", {})
            best_val = ckpt.get("best_val", None)
            
            # Calculate target epochs
            total_epochs = start_epoch + ADDITIONAL_EPOCHS
            
            print(f"✅ Checkpoint loaded successfully!")
            print()
            print("📊 CHECKPOINT INFORMATION:")
            print(f"   • Last completed epoch: {start_epoch}")
            print(f"   • Training will resume from: epoch {start_epoch + 1}")
            print(f"   • Target epoch: {total_epochs}")
            print(f"   • Additional epochs to train: {ADDITIONAL_EPOCHS}")
            
            if best_val is not None:
                print(f"   • Best validation loss so far: {best_val:.4f}")
            
            if 'loss' in saved_history and len(saved_history['loss']) > 0:
                print(f"   • Last training loss: {saved_history['loss'][-1]:.4f}")
                print(f"   • Last validation loss: {saved_history['val_loss'][-1]:.4f}")
            
            print()
            print("🔄 RESTORING MODEL STATE:")
            print("   ✓ Model weights")
            print("   ✓ Optimizer state (SGD with momentum)")
            print("   ✓ Learning rate scheduler (Cyclic LR)")
            print("   ✓ Training history")
            print("   ✓ Early stopping counter")
            if 'scaler_state' in ckpt:
                print("   ✓ AMP scaler state")
            print("-" * 70 + "\n")
            
            resume_path = CHECKPOINT_PATH
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            print()
            print("📌 FALLBACK: Starting NEW TRAINING from epoch 0")
            print(f"   Training will run for {NUM_EPOCHS} epochs")
            print("-" * 70 + "\n")
            
            resume_path = None
            total_epochs = NUM_EPOCHS
            start_epoch = 0
    
    # Execute training (new or continued depending on checkpoint availability)
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=total_epochs,
        patience=patience,
        base_lr=initial_lr,
        max_lr=max_lr,
        step_size=step_size,
        save_dir=CHECKPOINT_DIR,
        save_name=PYTORCH_CHECKPOINT_NAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        preview_gen=val_gen,
        resume_checkpoint=resume_path,  # None if checkpoint missing, else path
        history_pickle_path=PYTORCH_HISTORY_PATH,
    )
    
    print("\n" + "=" * 70)
    if resume_path is None:
        print("✅ NEW TRAINING COMPLETED (fallback mode)")
    else:
        print("✅ CONTINUED TRAINING COMPLETED")
        print(f"   Resumed from epoch: {start_epoch}")
    
    print(f"   Total epochs in history: {len(history['loss'])}")
    print(f"   Final training loss: {history['loss'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best validation loss: {min(history['val_loss']):.4f}")
    print(f"   Model checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70 + "\n")

# ================================
# TRAINING COMPLETE
# ================================

print("🎉 Training session finished!")
print(f"   Mode: {TRAINING_MODE.upper()}")
print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"   Model saved: {os.path.exists(CHECKPOINT_PATH)}")
print()
print("📈 Next steps:")
print("   1. Review training curves using plot_history_curves()")
print("   2. Evaluate model on test set")
print("   3. Generate predictions and visualizations")
print()

# ================================
# OPTIONAL: INTERACTIVE MODE
# ================================
# 
# For Colab notebook users who prefer interactive widgets, 
# uncomment the following section and place it BEFORE the 
# training execution logic:
#
# ```python
# # Install ipywidgets if not already installed
# try:
#     import ipywidgets as widgets
#     from IPython.display import display
#     HAS_WIDGETS = True
# except ImportError:
#     print("ipywidgets not available - using manual configuration")
#     HAS_WIDGETS = False
# 
# if HAS_WIDGETS:
#     print("=" * 70)
#     print("TRAINING MODE SELECTOR")
#     print("=" * 70)
#     
#     mode_widget = widgets.RadioButtons(
#         options=[
#             ('🔵 Start New Training (from epoch 0)', 'new'),
#             ('🟢 Continue Training (from checkpoint)', 'continue')
#         ],
#         value='new',
#         description='Mode:',
#         disabled=False,
#         style={'description_width': 'initial'}
#     )
#     
#     epochs_widget = widgets.IntText(
#         value=10000,
#         description='Total Epochs (new) / Additional Epochs (continue):',
#         disabled=False,
#         style={'description_width': 'initial'}
#     )
#     
#     checkpoint_widget = widgets.Text(
#         value=CHECKPOINT_PATH,
#         description='Checkpoint Path:',
#         disabled=False,
#         style={'description_width': 'initial'},
#         layout=widgets.Layout(width='80%')
#     )
#     
#     submit_button = widgets.Button(
#         description='Start Training',
#         button_style='success',
#         icon='check'
#     )
#     
#     output = widgets.Output()
#     
#     def on_submit_clicked(b):
#         with output:
#             output.clear_output()
#             TRAINING_MODE = mode_widget.value
#             NUM_EPOCHS = epochs_widget.value
#             ADDITIONAL_EPOCHS = epochs_widget.value
#             CHECKPOINT_PATH = checkpoint_widget.value
#             print(f"✅ Configuration saved!")
#             print(f"   Mode: {TRAINING_MODE}")
#             print(f"   Epochs: {epochs_widget.value}")
#             print(f"   Checkpoint: {CHECKPOINT_PATH}")
#             print()
#             print("Run the next cell to begin training.")
#     
#     submit_button.on_click(on_submit_clicked)
#     
#     display(mode_widget)
#     display(epochs_widget)
#     display(checkpoint_widget)
#     display(submit_button)
#     display(output)
#     
#     print()
#     print("👆 Select your training mode above, then run the next cell.")
#     print("=" * 70)
# ```

"""
HELPER FUNCTION: Check Training Status
"""

def check_training_status(checkpoint_path=None):
    """
    Utility function to inspect a checkpoint file without loading the model.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file. 
                        If None, uses default CHECKPOINT_PATH.
    
    Returns:
        dict: Dictionary containing checkpoint metadata
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': ckpt.get('epoch', 0),
            'best_val': ckpt.get('best_val', None),
            'no_improve': ckpt.get('no_improve', 0),
            'has_optimizer': 'optimizer_state' in ckpt,
            'has_scheduler': 'scheduler_state' in ckpt,
            'has_scaler': 'scaler_state' in ckpt,
            'has_history': 'history' in ckpt,
        }
        
        if 'history' in ckpt:
            hist = ckpt['history']
            info['total_epochs_trained'] = len(hist.get('loss', []))
            info['final_train_loss'] = hist['loss'][-1] if 'loss' in hist and len(hist['loss']) > 0 else None
            info['final_val_loss'] = hist['val_loss'][-1] if 'val_loss' in hist and len(hist['val_loss']) > 0 else None
            info['best_val_loss_achieved'] = min(hist['val_loss']) if 'val_loss' in hist and len(hist['val_loss']) > 0 else None
        
        print("=" * 70)
        print("CHECKPOINT STATUS")
        print("=" * 70)
        print(f"File: {checkpoint_path}")
        print(f"Last completed epoch: {info['epoch']}")
        print(f"Best validation loss: {info['best_val']:.4f}" if info['best_val'] else "Best validation loss: N/A")
        print(f"Epochs without improvement: {info['no_improve']}")
        print()
        print("Saved states:")
        print(f"  • Optimizer: {'✓' if info['has_optimizer'] else '✗'}")
        print(f"  • LR Scheduler: {'✓' if info['has_scheduler'] else '✗'}")
        print(f"  • AMP Scaler: {'✓' if info['has_scaler'] else '✗'}")
        print(f"  • History: {'✓' if info['has_history'] else '✗'}")
        
        if info['has_history']:
            print()
            print("Training history:")
            print(f"  • Total epochs: {info['total_epochs_trained']}")
            if info['final_train_loss']:
                print(f"  • Final train loss: {info['final_train_loss']:.4f}")
            if info['final_val_loss']:
                print(f"  • Final val loss: {info['final_val_loss']:.4f}")
            if info['best_val_loss_achieved']:
                print(f"  • Best val loss ever: {info['best_val_loss_achieved']:.4f}")
        
        print("=" * 70)
        
        return info
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return None

# Example usage:
# status = check_training_status()
# status = check_training_status("/path/to/custom/checkpoint.pt")
