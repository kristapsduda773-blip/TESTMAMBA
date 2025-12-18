"""
EXAMPLE: Properly Structured Training Notebook for Google Colab
================================================================

This file demonstrates the correct structure for a Colab notebook with
user-selectable training mode (new vs. continue).

Copy the relevant sections into separate notebook cells.
"""

# ============================================================================
# CELL 1: Configuration (User edits this cell)
# ============================================================================

# ========================================
# USER CONFIGURATION - EDIT THIS SECTION
# ========================================

# TRAINING MODE SELECTION
TRAINING_MODE = "new"  # Options: "new" or "continue"
#
# "new"      → Start from epoch 0, fresh weights
# "continue" → Resume from last checkpoint

# DATASET PATHS
# Update this to match your Google Drive structure
DATASET_BASE_PATH = "/content/drive/MyDrive/BHSD/SPLIT/segmentation"

# Alternative if your data is in "Computers" folder:
# DATASET_BASE_PATH = "/content/drive/Othercomputers/My Laptop/BHSD/SPLIT/segmentation"

# CHECKPOINT CONFIGURATION
CHECKPOINT_DIR = '/content/drive/MyDrive/Unet_checkpoints'
CHECKPOINT_FILENAME = "best_model_mamba_pytorch.pt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_FILENAME}"

# TRAINING HYPERPARAMETERS
NUM_EPOCHS = 10000           # Total epochs for new training
ADDITIONAL_EPOCHS = 1000     # Additional epochs when continuing
PATIENCE = 30                # Early stopping patience
INITIAL_LR = 1e-5            # Minimum learning rate (cyclic)
MAX_LR = 1e-2                # Maximum learning rate (cyclic)
STEP_SIZE = 670              # CLR half-cycle length

# CLASS WEIGHTS (calculated from your dataset)
CLASS_WEIGHT_TUPLE = (0.168, 931.755, 71.288, 249.718, 275.719, 136.199)

# ========================================
# END OF USER CONFIGURATION
# ========================================

print("=" * 70)
print("CONFIGURATION LOADED")
print("=" * 70)
print(f"Training Mode:    {TRAINING_MODE.upper()}")
print(f"Dataset Path:     {DATASET_BASE_PATH}")
print(f"Checkpoint Dir:   {CHECKPOINT_DIR}")
print(f"Checkpoint File:  {CHECKPOINT_FILENAME}")
print(f"Epochs (new):     {NUM_EPOCHS}")
print(f"Epochs (add):     {ADDITIONAL_EPOCHS}")
print(f"Early Stop:       {PATIENCE} epochs")
print(f"Learning Rate:    {INITIAL_LR} → {MAX_LR} (cyclic)")
print("=" * 70)


# ============================================================================
# CELL 2: Environment Setup & Imports
# ============================================================================

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✓ Google Drive mounted")
except ImportError:
    IN_COLAB = False
    print("⚠️  Not running in Colab")

# Install dependencies
import subprocess
import sys
import os

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

# Check and install required packages
required_packages = {
    'torch': 'torch',
    'nibabel': 'nibabel',
    'patchify': 'patchify',
    'mamba_ssm': 'mamba-ssm',
}

print("\nChecking dependencies...")
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"⚙️  Installing {package_name}...")
        install_package(package_name)
        print(f"✓ {package_name} installed")

# Import libraries
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
from patchify import patchify
from tqdm import tqdm
import copy
import time

print(f"\n✓ All imports successful")
print(f"  NumPy version: {np.__version__}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# CELL 3: Verify Dataset & Paths
# ============================================================================

print("\nVerifying paths...")

# Check dataset exists
if not os.path.exists(DATASET_BASE_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found at: {DATASET_BASE_PATH}\n"
        f"Please update DATASET_BASE_PATH in Cell 1"
    )
print(f"✓ Dataset found: {DATASET_BASE_PATH}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"✓ Checkpoint directory ready: {CHECKPOINT_DIR}")

# Define dataset subdirectories
SEG_TRAIN_IMG = os.path.join(DATASET_BASE_PATH, "train_images")
SEG_TRAIN_MASK = os.path.join(DATASET_BASE_PATH, "train_masks")
SEG_VAL_IMG = os.path.join(DATASET_BASE_PATH, "val_images")
SEG_VAL_MASK = os.path.join(DATASET_BASE_PATH, "val_masks")
SEG_TEST_IMG = os.path.join(DATASET_BASE_PATH, "test_images")
SEG_TEST_MASK = os.path.join(DATASET_BASE_PATH, "test_masks")

# Verify all subdirectories exist
for name, path in [
    ("Train images", SEG_TRAIN_IMG),
    ("Train masks", SEG_TRAIN_MASK),
    ("Val images", SEG_VAL_IMG),
    ("Val masks", SEG_VAL_MASK),
    ("Test images", SEG_TEST_IMG),
    ("Test masks", SEG_TEST_MASK),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} not found at: {path}")
    count = len([f for f in os.listdir(path) if f.endswith('.nii') or f.endswith('.nii.gz')])
    print(f"✓ {name:15s}: {count:3d} files")

print("\n✓ All paths verified")


# ============================================================================
# CELL 4: Data Generator
# ============================================================================

# Insert your PatchGenerator class here
# This is a placeholder - use your actual implementation

class PatchGenerator:
    """
    Data generator for 3D patch extraction from medical images.
    (Use your actual implementation here)
    """
    def __init__(self, image_dir, mask_dir, patch_size=(256,256,8), 
                 step_size=(256,256,4), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.step_size = step_size
        self.augment = augment
        # ... (rest of your implementation)
        pass
    
    def __len__(self):
        # Return number of patches
        pass
    
    def __getitem__(self, idx):
        # Return (image_patch, mask_patch)
        pass

print("✓ Data generator defined")


# ============================================================================
# CELL 5: Create Data Generators
# ============================================================================

print("Creating data generators...")

train_gen = PatchGenerator(
    SEG_TRAIN_IMG, 
    SEG_TRAIN_MASK,
    patch_size=(256, 256, 8),
    step_size=(256, 256, 4),
    augment=True
)

val_gen = PatchGenerator(
    SEG_VAL_IMG,
    SEG_VAL_MASK,
    patch_size=(256, 256, 8),
    step_size=(256, 256, 4),
    augment=False
)

test_gen = PatchGenerator(
    SEG_TEST_IMG,
    SEG_TEST_MASK,
    patch_size=(256, 256, 8),
    step_size=(256, 256, 4),
    augment=False
)

print(f"✓ Train generator: {len(train_gen)} patches")
print(f"✓ Val generator:   {len(val_gen)} patches")
print(f"✓ Test generator:  {len(test_gen)} patches")


# ============================================================================
# CELL 6: Model Architecture
# ============================================================================

# Insert your model definition here
# This is a placeholder - use your actual UNet3DMamba implementation

class UNet3DMamba(nn.Module):
    """
    3D U-Net with Tri-Oriented Mamba blocks.
    (Use your actual implementation here)
    """
    def __init__(self, in_channels=4, num_classes=6, base_filters=32):
        super().__init__()
        # ... (your model architecture)
        pass
    
    def forward(self, x):
        # ... (your forward pass)
        pass

print("✓ Model architecture defined")


# ============================================================================
# CELL 7: Loss Functions & Metrics
# ============================================================================

# Insert your loss function implementations
# These are placeholders - use your actual implementations

def generalized_dice_loss(y_true, y_pred, class_weights):
    """GDL implementation"""
    pass

def weighted_categorical_crossentropy(y_true, y_pred, class_weights):
    """WCE implementation"""
    pass

def combined_loss_pt(y_true, y_pred, class_weights, alpha=0.7):
    """Combined loss: alpha*GDL + (1-alpha)*WCE"""
    gdl = generalized_dice_loss(y_true, y_pred, class_weights)
    wce = weighted_categorical_crossentropy(y_true, y_pred, class_weights)
    return alpha * gdl + (1 - alpha) * wce

def dice_metric_pt(y_true, y_pred, class_weights):
    """Dice coefficient metric"""
    pass

print("✓ Loss functions defined")


# ============================================================================
# CELL 8: Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, device, class_weights, 
                alpha=0.7, scaler=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    count = 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with AMP
        use_amp = scaler is not None
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            logits = model(x)
            loss = combined_loss_pt(y, logits, class_weights, alpha=alpha)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Metrics
        with torch.no_grad():
            dice = dice_metric_pt(y, logits, class_weights)
        
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        count += 1
    
    return epoch_loss / max(1, count), epoch_dice / max(1, count)


def eval_epoch(model, loader, device, class_weights):
    """Evaluate for one epoch"""
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    count = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = combined_loss_pt(y, logits, class_weights)
            dice = dice_metric_pt(y, logits, class_weights)
            
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            count += 1
    
    return epoch_loss / max(1, count), epoch_dice / max(1, count)


print("✓ Training functions defined")


# ============================================================================
# CELL 9: Main Training Loop
# ============================================================================

def fit_pytorch_mamba(
    train_gen,
    val_gen,
    num_epochs=10000,
    patience=30,
    base_lr=1e-5,
    max_lr=1e-2,
    step_size=670,
    save_dir='/content/drive/MyDrive/Unet_checkpoints',
    save_name="best_model.pt",
    class_weights_list=(1.0,),
    device_str=None,
    resume_checkpoint=None,
):
    """
    Main training loop with checkpoint saving/loading.
    """
    # Setup
    device = torch.device(device_str or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Build model
    model = UNet3DMamba(in_channels=4, num_classes=6, base_filters=32).to(device)
    
    # Class weights
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32, device=device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=base_lr, max_lr=max_lr, 
        step_size_up=step_size, mode='triangular'
    )
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training state
    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'loss': [], 'val_loss': [], 'dice': [], 'val_dice': [], 'lr': []}
    start_epoch = 0
    
    # Load checkpoint if resuming
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        
        model.load_state_dict(ckpt['model_state'])
        best_state = copy.deepcopy(model.state_dict())
        start_epoch = ckpt.get('epoch', 0)
        history = ckpt.get('history', history)
        best_val = ckpt.get('best_val', best_val)
        no_improve = ckpt.get('no_improve', 0)
        
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if scaler and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch+1} to {num_epochs}")
    print("=" * 70)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, class_weights, scaler=scaler
        )
        
        # Validate
        val_loss, val_dice = eval_epoch(model, val_loader, device, class_weights)
        
        # Record
        current_lr = scheduler.get_last_lr()[0]
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Print progress
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1:5d}/{num_epochs}  "
              f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"dice={train_dice:.4f}  val_dice={val_dice:.4f}  "
              f"lr={current_lr:.6f}  time={elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            
            checkpoint = {
                'model_state': best_state,
                'epoch': epoch + 1,
                'history': history,
                'best_val': best_val,
                'no_improve': 0,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
            }
            if scaler:
                checkpoint['scaler_state'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(save_dir, save_name))
            no_improve = 0
            print(f"  ✓ Best model saved (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"  Epochs without improvement: {no_improve}/{patience}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


print("✓ Main training function defined")


# ============================================================================
# CELL 10: Execute Training (User runs this cell)
# ============================================================================

# Validate configuration
if TRAINING_MODE not in ["new", "continue"]:
    raise ValueError(f"Invalid TRAINING_MODE: {TRAINING_MODE}. Use 'new' or 'continue'")

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

# Build data loaders (PyTorch DataLoader)
from torch.utils.data import DataLoader

train_loader = DataLoader(train_gen, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, num_workers=2)

# Execute based on mode
if TRAINING_MODE == "new":
    print("\n🔵 MODE: NEW TRAINING")
    print("Starting from epoch 0 with fresh weights\n")
    
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        base_lr=INITIAL_LR,
        max_lr=MAX_LR,
        step_size=STEP_SIZE,
        save_dir=CHECKPOINT_DIR,
        save_name=CHECKPOINT_FILENAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        resume_checkpoint=None,  # Start fresh
    )
    
elif TRAINING_MODE == "continue":
    print("\n🟢 MODE: CONTINUE TRAINING")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️  Checkpoint not found: {CHECKPOINT_PATH}")
        print("Falling back to NEW TRAINING\n")
        resume_path = None
        total_epochs = NUM_EPOCHS
    else:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        total_epochs = start_epoch + ADDITIONAL_EPOCHS
        resume_path = CHECKPOINT_PATH
        print(f"Resuming from epoch {start_epoch} → {total_epochs}\n")
    
    model, history = fit_pytorch_mamba(
        train_gen=train_gen,
        val_gen=val_gen,
        num_epochs=total_epochs,
        patience=PATIENCE,
        base_lr=INITIAL_LR,
        max_lr=MAX_LR,
        step_size=STEP_SIZE,
        save_dir=CHECKPOINT_DIR,
        save_name=CHECKPOINT_FILENAME,
        class_weights_list=CLASS_WEIGHT_TUPLE,
        device_str=None,
        resume_checkpoint=resume_path,
    )

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE")
print("=" * 70)
print(f"Total epochs: {len(history['loss'])}")
print(f"Best val loss: {min(history['val_loss']):.4f}")
print(f"Model saved: {CHECKPOINT_PATH}")
print("=" * 70)


# ============================================================================
# CELL 11: Plot Training Curves
# ============================================================================

def plot_training_curves(history):
    """Plot loss, dice, and learning rate curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice
    axes[1].plot(epochs, history['dice'], label='Train Dice')
    axes[1].plot(epochs, history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Metric per Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], label='Learning Rate', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot results
plot_training_curves(history)


# ============================================================================
# CELL 12: Model Evaluation (Optional)
# ============================================================================

print("Evaluating on test set...")

test_loader = DataLoader(test_gen, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.tensor(CLASS_WEIGHT_TUPLE, device=device)

test_loss, test_dice = eval_epoch(model, test_loader, device, class_weights)

print(f"\n{'=' * 70}")
print("TEST SET RESULTS")
print(f"{'=' * 70}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice: {test_dice:.4f}")
print(f"{'=' * 70}")

# ============================================================================
# END OF EXAMPLE
# ============================================================================
