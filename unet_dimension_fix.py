"""
Fix for U-Net Dimension Mismatch Error
=======================================

The error occurs because skip connections don't align after upsampling.
This happens when input dimensions don't divide evenly by 2^num_downsamples.

Solution: Match spatial dimensions dynamically during forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# OPTION 1: Fix the forward pass with size matching
# =====================================================================

def forward_with_size_matching(self, x: torch.Tensor) -> torch.Tensor:
    """
    Fixed forward pass that matches skip connection sizes dynamically.
    Replace the forward method in UNet3DMamba with this.
    """
    # Encoder path
    x1 = self.m1(self.enc_in(x))              # (N, f32, D, H, W)
    x2_in = self.down1(x1)                    # (N, f64, D, H/2, W/2)
    x2 = self.m2(x2_in)
    x3_in = self.down2(x2)                    # (N, f128, D, H/4, W/4)
    x3 = self.m3(x3_in)
    x4_in = self.down3(x3)                    # (N, f256, D, H/8, W/8)
    x4 = self.m4(x4_in)
    x5_in = self.down4(x4)                    # (N, f512, D, H/16, W/16)
    x5 = self.m5(x5_in)
    base = self.base(x5)                      # (N, f512*2, D, H/32, W/32)

    # Decoder path with size matching
    u = F.interpolate(base, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = self.up1(u)
    # Match x5 size
    if u.shape != x5.shape:
        u = F.interpolate(u, size=x5.shape[2:], mode='trilinear', align_corners=False)
    u = torch.cat([u, x5], dim=1)
    u = self.dec1(u)

    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = self.up2(u)
    # Match x4 size
    if u.shape != x4.shape:
        u = F.interpolate(u, size=x4.shape[2:], mode='trilinear', align_corners=False)
    u = torch.cat([u, x4], dim=1)
    u = self.dec2(u)

    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = self.up3(u)
    # Match x3 size
    if u.shape != x3.shape:
        u = F.interpolate(u, size=x3.shape[2:], mode='trilinear', align_corners=False)
    u = torch.cat([u, x3], dim=1)
    u = self.dec3(u)

    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = self.up4(u)
    # Match x2 size
    if u.shape != x2.shape:
        u = F.interpolate(u, size=x2.shape[2:], mode='trilinear', align_corners=False)
    u = torch.cat([u, x2], dim=1)
    u = self.dec4(u)

    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = self.up5(u)
    # Match x1 size
    if u.shape != x1.shape:
        u = F.interpolate(u, size=x1.shape[2:], mode='trilinear', align_corners=False)
    u = torch.cat([u, x1], dim=1)
    u = self.dec5(u)

    logits = self.out_conv(u)  # (N, num_classes, D, H, W)
    return logits


# =====================================================================
# OPTION 2: Input validation and padding
# =====================================================================

def ensure_divisible_size(tensor, divisor=32):
    """
    Pad tensor to ensure spatial dimensions are divisible by divisor.
    
    Args:
        tensor: Input tensor (N, C, D, H, W)
        divisor: Required divisor (default 32 for 5 downsampling layers)
    
    Returns:
        padded_tensor, padding_info
    """
    N, C, D, H, W = tensor.shape
    
    # Calculate required padding
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    
    if pad_h == 0 and pad_w == 0:
        return tensor, None
    
    # Pad: (left, right, top, bottom, front, back)
    padding = (0, pad_w, 0, pad_h, 0, 0)
    padded = F.pad(tensor, padding, mode='constant', value=0)
    
    padding_info = {'original_shape': (N, C, D, H, W), 'padding': padding}
    return padded, padding_info


def remove_padding(tensor, padding_info):
    """Remove padding added by ensure_divisible_size."""
    if padding_info is None:
        return tensor
    
    N, C, D, H, W = padding_info['original_shape']
    return tensor[:, :, :D, :H, :W]


# =====================================================================
# OPTION 3: Diagnostic function
# =====================================================================

def diagnose_shape_mismatch(model, input_shape=(1, 4, 8, 256, 256)):
    """
    Run a forward pass and print all intermediate shapes.
    Helps identify where the dimension mismatch occurs.
    """
    device = next(model.parameters()).device
    x = torch.randn(*input_shape).to(device)
    
    print("=" * 70)
    print("SHAPE DIAGNOSIS")
    print("=" * 70)
    print(f"Input shape: {x.shape}")
    print()
    
    # Encoder
    x1 = model.m1(model.enc_in(x))
    print(f"x1 (enc level 1): {x1.shape}")
    
    x2_in = model.down1(x1)
    x2 = model.m2(x2_in)
    print(f"x2 (enc level 2): {x2.shape}")
    
    x3_in = model.down2(x2)
    x3 = model.m3(x3_in)
    print(f"x3 (enc level 3): {x3.shape}")
    
    x4_in = model.down3(x3)
    x4 = model.m4(x4_in)
    print(f"x4 (enc level 4): {x4.shape}")
    
    x5_in = model.down4(x4)
    x5 = model.m5(x5_in)
    print(f"x5 (enc level 5): {x5.shape}")
    
    base = model.base(x5)
    print(f"base (bottleneck): {base.shape}")
    print()
    
    # Decoder
    u = F.interpolate(base, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = model.up1(u)
    print(f"u after up1: {u.shape} | x5: {x5.shape} | Match: {u.shape[2:] == x5.shape[2:]}")
    
    u = torch.cat([u, x5], dim=1)
    u = model.dec1(u)
    
    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = model.up2(u)
    print(f"u after up2: {u.shape} | x4: {x4.shape} | Match: {u.shape[2:] == x4.shape[2:]}")
    
    u = torch.cat([u, x4], dim=1)
    u = model.dec2(u)
    
    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = model.up3(u)
    print(f"u after up3: {u.shape} | x3: {x3.shape} | Match: {u.shape[2:] == x3.shape[2:]}")
    
    u = torch.cat([u, x3], dim=1)
    u = model.dec3(u)
    
    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = model.up4(u)
    print(f"u after up4: {u.shape} | x2: {x2.shape} | Match: {u.shape[2:] == x2.shape[2:]}")
    
    u = torch.cat([u, x2], dim=1)
    u = model.dec4(u)
    
    u = F.interpolate(u, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    u = model.up5(u)
    print(f"u after up5: {u.shape} | x1: {x1.shape} | Match: {u.shape[2:] == x1.shape[2:]}")
    
    print("=" * 70)


# =====================================================================
# USAGE EXAMPLES
# =====================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION INSTRUCTIONS")
    print("=" * 70)
    
    print("""
    STEP 1: Diagnose the problem
    ----------------------------
    Add this to your notebook:
    
        from unet_dimension_fix import diagnose_shape_mismatch
        diagnose_shape_mismatch(model, input_shape=(1, 4, 8, 256, 256))
    
    This will show you exactly where dimensions don't match.
    
    
    STEP 2: Apply the fix
    ---------------------
    
    OPTION A - Quick fix (recommended):
    Replace the forward method in your UNet3DMamba class (line ~1713)
    Copy the forward_with_size_matching function from this file.
    
    OPTION B - Input padding:
    Wrap your model calls with padding:
        
        x_padded, pad_info = ensure_divisible_size(x, divisor=32)
        output_padded = model(x_padded)
        output = remove_padding(output_padded, pad_info)
    
    
    STEP 3: Verify the fix
    ----------------------
    Run smoke test:
        
        x_test = torch.randn(1, 4, 8, 256, 256).to(device)
        y_test = model(x_test)
        print(f"Input: {x_test.shape} -> Output: {y_test.shape}")
        # Should print: Input: torch.Size([1, 4, 8, 256, 256]) -> Output: torch.Size([1, 6, 8, 256, 256])
    """)
