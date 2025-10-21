"""
Python implementation of ImageMagick's LUT (Look-Up Table) application methods.

This module provides two LUT application methods:
1. clut_image: Simple 1D LUT mapping using a gradient/strip LUT image
2. hald_clut_image: 3D Hald CLUT mapping using a Hald CLUT image

Based on ImageMagick's MagickCore/enhance.c implementation.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
from enum import Enum


class InterpolationMethod(Enum):
    """Interpolation methods for LUT sampling."""
    NEAREST = 0
    BILINEAR = 1


def clut_image(
    src_image: np.ndarray,
    lut_image: np.ndarray,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
) -> np.ndarray:
    """
    Apply a 1D CLUT (Color Look-Up Table) to an image.
    
    This function applies a color lookup table to transform the colors in the source image.
    Each channel value (R, G, B, A) is used as an index to look up a new value from the LUT.
    
    The LUT image is sampled to create a lookup table with 256 entries. For each pixel in the
    source image, the channel values (0-255) are used as indices into this lookup table to
    determine the output color.
    
    Args:
        src_image: Source image as numpy array (H, W, C) with values in range [0, 255].
                   Can be RGB (3 channels) or RGBA (4 channels). dtype should be uint8.
        lut_image: LUT image as numpy array (H, W, C) with values in range [0, 255].
                   The LUT is sampled across its width and height to build the lookup table.
                   dtype should be uint8.
        interpolation: Interpolation method for sampling the LUT image.
    
    Returns:
        Transformed image as numpy array with same shape as src_image.
    
    Example:
        >>> src = np.array(Image.open('source.png'))
        >>> lut = np.array(Image.open('lut.png'))
        >>> result = clut_image(src, lut)
        >>> Image.fromarray(result).save('output.png')
    """
    if src_image.dtype != np.uint8:
        raise ValueError("Source image must be uint8 dtype")
    if lut_image.dtype != np.uint8:
        raise ValueError("LUT image must be uint8 dtype")
    
    # MaxMap in ImageMagick is 255 for 8-bit images
    MAX_MAP = 255
    
    # Ensure we have at least 3 channels (RGB)
    if src_image.ndim == 2:
        src_image = np.stack([src_image] * 3, axis=-1)
    if lut_image.ndim == 2:
        lut_image = np.stack([lut_image] * 3, axis=-1)
    
    src_channels = src_image.shape[2] if src_image.ndim == 3 else 1
    lut_channels = lut_image.shape[2] if lut_image.ndim == 3 else 1
    
    # Build the CLUT map by sampling the LUT image
    # ImageMagick samples the LUT image across both width and height
    clut_map = np.zeros((MAX_MAP + 1, max(src_channels, lut_channels)), dtype=np.float32)
    
    lut_height, lut_width = lut_image.shape[:2]
    
    # Adjustment for interpolation (0 for nearest, 1 for bilinear)
    adjust = 0 if interpolation == InterpolationMethod.NEAREST else 1
    
    # Sample the LUT image to build the lookup table
    for i in range(MAX_MAP + 1):
        # Calculate position in LUT image (normalized to [0, 1])
        x = i * (lut_width - adjust) / MAX_MAP
        y = i * (lut_height - adjust) / MAX_MAP
        
        if interpolation == InterpolationMethod.NEAREST:
            # Nearest neighbor interpolation
            xi = int(np.clip(np.round(x), 0, lut_width - 1))
            yi = int(np.clip(np.round(y), 0, lut_height - 1))
            clut_map[i, :lut_channels] = lut_image[yi, xi].astype(np.float32)
        else:
            # Bilinear interpolation
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, lut_width - 1)
            y1 = min(y0 + 1, lut_height - 1)
            
            x0 = max(0, x0)
            y0 = max(0, y0)
            
            fx = x - x0
            fy = y - y0
            
            # Bilinear interpolation formula
            c00 = lut_image[y0, x0].astype(np.float32)
            c01 = lut_image[y0, x1].astype(np.float32)
            c10 = lut_image[y1, x0].astype(np.float32)
            c11 = lut_image[y1, x1].astype(np.float32)
            
            interpolated = (
                c00 * (1 - fx) * (1 - fy) +
                c01 * fx * (1 - fy) +
                c10 * (1 - fx) * fy +
                c11 * fx * fy
            )
            clut_map[i, :lut_channels] = interpolated
    
    # Apply the CLUT to the source image
    # Each channel value is used as an index into the clut_map
    output = np.zeros_like(src_image, dtype=np.uint8)
    
    for c in range(min(src_channels, lut_channels)):
        # Use source pixel values as indices into the lookup table
        indices = src_image[:, :, c]
        output[:, :, c] = np.clip(clut_map[indices, c], 0, 255).astype(np.uint8)
    
    # Copy alpha channel if present and not in LUT
    if src_channels == 4:
        if lut_channels == 4:
            indices = src_image[:, :, 3]
            output[:, :, 3] = np.clip(clut_map[indices, 3], 0, 255).astype(np.uint8)
        else:
            output[:, :, 3] = src_image[:, :, 3]
    
    return output


def hald_clut_image(
    src_image: np.ndarray,
    hald_image: np.ndarray,
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
) -> np.ndarray:
    """
    Apply a 3D Hald CLUT (Color Look-Up Table) to an image.
    
    A Hald CLUT is a 3D color lookup table stored as a 2D image. The RGB values of each
    pixel in the source image are used as 3D coordinates to look up the corresponding
    color in the Hald CLUT image.
    
    Hald CLUTs are typically square images where the size is level^2 x level^2, where
    level = cube_root(min(width, height)). Common sizes are:
    - Level 8: 64x64 pixels (8x8x8 color cube)
    - Level 16: 256x256 pixels (16x16x16 color cube)
    - Level 32: 1024x1024 pixels (32x32x32 color cube)
    
    Args:
        src_image: Source image as numpy array (H, W, C) with values in range [0, 255].
                   Must be RGB (3 channels) or RGBA (4 channels). dtype should be uint8.
        hald_image: Hald CLUT image as numpy array (H, W, C) with values in range [0, 255].
                    Must be a valid Hald CLUT image. dtype should be uint8.
        interpolation: Interpolation method for sampling the Hald CLUT.
    
    Returns:
        Transformed image as numpy array with same shape as src_image.
    
    Example:
        >>> src = np.array(Image.open('source.png'))
        >>> hald = np.array(Image.open('hald_clut.png'))
        >>> result = hald_clut_image(src, hald)
        >>> Image.fromarray(result).save('output.png')
    """
    if src_image.dtype != np.uint8:
        raise ValueError("Source image must be uint8 dtype")
    if hald_image.dtype != np.uint8:
        raise ValueError("Hald image must be uint8 dtype")
    
    # Ensure we have at least 3 channels (RGB)
    if src_image.ndim == 2:
        src_image = np.stack([src_image] * 3, axis=-1)
    if hald_image.ndim == 2:
        hald_image = np.stack([hald_image] * 3, axis=-1)
    
    has_alpha = src_image.shape[2] == 4
    
    hald_height, hald_width = hald_image.shape[:2]
    
    # Calculate the level of the Hald CLUT
    # level^2 * level^2 = width (approximately)
    length = min(hald_width, hald_height)
    level = 2
    while (level * level * level) < length:
        level += 1
    level *= level
    cube_size = level * level
    
    # Convert source image to float in range [0, 1]
    src_float = src_image[:, :, :3].astype(np.float32) / 255.0
    
    # Extract RGB channels
    r = src_float[:, :, 0] * (level - 1)
    g = src_float[:, :, 1] * (level - 1)
    b = src_float[:, :, 2] * (level - 1)
    
    # Calculate offset in the Hald image
    # The Hald CLUT lays out a 3D cube in 2D: x + level*floor(y) + cube_size*floor(z)
    offset = r + level * np.floor(g) + cube_size * np.floor(b)
    
    # Calculate fractional parts for interpolation
    r_frac = r - np.floor(r)
    g_frac = g - np.floor(g)
    b_frac = b - np.floor(b)
    
    # Convert offset to 2D coordinates
    x = np.fmod(offset, hald_width)
    y = np.floor(offset / hald_width)
    
    # Sample 8 points of the 3D cube for trilinear interpolation
    def sample_hald(x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
        """Sample the Hald image at the given coordinates with interpolation."""
        x_coord = np.clip(x_coord, 0, hald_width - 1)
        y_coord = np.clip(y_coord, 0, hald_height - 1)
        
        if interpolation == InterpolationMethod.NEAREST:
            xi = np.round(x_coord).astype(int)
            yi = np.round(y_coord).astype(int)
            return hald_image[yi, xi, :3].astype(np.float32)
        else:
            x0 = np.floor(x_coord).astype(int)
            y0 = np.floor(y_coord).astype(int)
            x1 = np.clip(x0 + 1, 0, hald_width - 1)
            y1 = np.clip(y0 + 1, 0, hald_height - 1)
            
            fx = x_coord - x0
            fy = y_coord - y0
            
            # Expand dimensions for broadcasting
            fx = fx[:, :, np.newaxis]
            fy = fy[:, :, np.newaxis]
            
            c00 = hald_image[y0, x0, :3].astype(np.float32)
            c01 = hald_image[y0, x1, :3].astype(np.float32)
            c10 = hald_image[y1, x0, :3].astype(np.float32)
            c11 = hald_image[y1, x1, :3].astype(np.float32)
            
            return (
                c00 * (1 - fx) * (1 - fy) +
                c01 * fx * (1 - fy) +
                c10 * (1 - fx) * fy +
                c11 * fx * fy
            )
    
    # Sample the two slices in the green direction and blend
    offset1 = offset
    offset2 = offset + level
    
    x1 = np.fmod(offset1, hald_width)
    y1 = np.floor(offset1 / hald_width)
    x2 = np.fmod(offset2, hald_width)
    y2 = np.floor(offset2 / hald_width)
    
    pixel1 = sample_hald(x1, y1)
    pixel2 = sample_hald(x2, y2)
    
    # Blend in green direction
    if interpolation == InterpolationMethod.NEAREST:
        area_g = (g_frac >= 0.5).astype(np.float32)
    else:
        area_g = g_frac
    area_g = area_g[:, :, np.newaxis]
    
    pixel_g1 = pixel1 * (1 - area_g) + pixel2 * area_g
    
    # Sample the two slices in the blue direction and blend
    offset3 = offset + cube_size
    offset4 = offset + cube_size + level
    
    x3 = np.fmod(offset3, hald_width)
    y3 = np.floor(offset3 / hald_width)
    x4 = np.fmod(offset4, hald_width)
    y4 = np.floor(offset4 / hald_width)
    
    pixel3 = sample_hald(x3, y3)
    pixel4 = sample_hald(x4, y4)
    
    # Blend in green direction for the second blue slice
    pixel_g2 = pixel3 * (1 - area_g) + pixel4 * area_g
    
    # Blend in blue direction
    if interpolation == InterpolationMethod.NEAREST:
        area_b = (b_frac >= 0.5).astype(np.float32)
    else:
        area_b = b_frac
    area_b = area_b[:, :, np.newaxis]
    
    pixel_final = pixel_g1 * (1 - area_b) + pixel_g2 * area_b
    
    # Clamp to valid range and convert to uint8
    output = np.clip(pixel_final, 0, 255).astype(np.uint8)
    
    # Preserve alpha channel if present
    if has_alpha:
        result = np.zeros_like(src_image)
        result[:, :, :3] = output
        result[:, :, 3] = src_image[:, :, 3]
        return result
    
    return output


def apply_lut_from_files(
    src_path: str,
    lut_path: str,
    output_path: str,
    method: str = "auto",
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
) -> None:
    """
    Apply a LUT from image files.
    
    Args:
        src_path: Path to source image
        lut_path: Path to LUT image
        output_path: Path to save output image
        method: LUT method to use: "clut", "hald", or "auto" (default).
                "auto" will use hald for square images, clut otherwise.
        interpolation: Interpolation method
    """
    # Load images
    src_img = Image.open(src_path).convert('RGB')
    lut_img = Image.open(lut_path).convert('RGB')
    
    # Convert to numpy arrays
    src_array = np.array(src_img)
    lut_array = np.array(lut_img)
    
    # Determine method
    if method == "auto":
        lut_height, lut_width = lut_array.shape[:2]
        # Square images are likely Hald CLUTs
        if abs(lut_width - lut_height) < 10:
            # Check if it's a valid Hald size
            length = min(lut_width, lut_height)
            level = 2
            while (level * level * level) < length:
                level += 1
            # If it's close to a perfect Hald size, use hald method
            expected_size = level * level
            if abs(length - expected_size) < 10:
                method = "hald"
            else:
                method = "clut"
        else:
            method = "clut"
    
    # Apply LUT
    if method == "hald":
        result = hald_clut_image(src_array, lut_array, interpolation)
    else:
        result = clut_image(src_array, lut_array, interpolation)
    
    # Save result
    result_img = Image.fromarray(result)
    result_img.save(output_path)
    print(f"Applied {method.upper()} LUT and saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python lut_apply.py <source_image> <lut_image> <output_image> [method]")
        print("  method: 'clut', 'hald', or 'auto' (default)")
        print()
        print("Example:")
        print("  python lut_apply.py photo.jpg my_lut.png output.jpg auto")
        sys.exit(1)
    
    src_path = sys.argv[1]
    lut_path = sys.argv[2]
    output_path = sys.argv[3]
    method = sys.argv[4] if len(sys.argv) > 4 else "auto"
    
    apply_lut_from_files(src_path, lut_path, output_path, method)

