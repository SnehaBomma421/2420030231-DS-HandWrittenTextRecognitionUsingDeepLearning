import cv2
import numpy as np


def read_image(file_bytes):
    """
    Convert uploaded image bytes into an OpenCV BGR image.
    Handles orientation and color channel integrity.
    """
    if file_bytes is None:
        return None
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def remove_shadows(gray):
    """
    Remove uneven lighting gradients, shadows, and phone camera falloff
    via scale-adaptive morphological background illumination division.
    Produces a normalized paper background with consistent luminance without smudging strokes.
    """
    h, w = gray.shape[:2]
    # Scale kernel size to image dimensions so small notes don't get over-blurred
    k_size = max(9, min(25, int(min(h, w) * 0.25)))
    if k_size % 2 == 0:
        k_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    dilated = cv2.dilate(gray, kernel)
    bg = cv2.medianBlur(dilated, k_size)

    # Normalized division: (gray / bg) * 255
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def suppress_bleed_through_and_binarize(norm, offset=14):
    """
    Suppresses faint reverse-side pen bleed-through and paper impressions
    while keeping handwriting strokes and numbers crisp and sharp.
    Uses scale-adaptive block sizing to prevent character blooming/smudging.
    """
    h, w = norm.shape[:2]

    # 1. Gentle CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # 2. Light Gaussian blur (3, 3) to smooth paper grain without thickening strokes
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 3. Scale-adaptive block size
    # Small images get small blocks; large scans get proportional blocks
    block_size = max(11, min(41, int(min(h, w) * 0.12)))
    if block_size % 2 == 0:
        block_size += 1

    # 4. Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        offset
    )

    return binary, enhanced


def remove_ruled_lines(binary, min_line_len=30):
    """
    Detect and remove horizontal notebook ruled lines and vertical margin lines
    from the binary segmentation mask while strictly preserving handwriting strokes
    and numbers that cross or touch them.
    
    Uses directional stroke protection without character-fusing closing morphology.
    """
    h, w = binary.shape
    h_len = max(28, int(w * 0.035))
    v_len = max(28, int(h * 0.035))

    # Gentle 4px bridge for small breaks in ruled lines without fusing characters
    bridged_h = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    )
    bridged_v = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    )

    # Extract horizontal ruled lines
    h_lines = cv2.morphologyEx(
        bridged_h,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    )

    # Extract vertical margin lines
    v_lines = cv2.morphologyEx(
        bridged_v,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    )

    # 1. Vertical stroke protection for horizontal lines (preserves letter stems & digits)
    v_strokes = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    )
    v_prot = cv2.dilate(
        v_strokes,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    )
    h_lines_to_remove = cv2.bitwise_and(h_lines, cv2.bitwise_not(v_prot))

    # 2. Horizontal stroke protection for vertical margin lines
    h_strokes = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    )
    h_prot = cv2.dilate(
        h_strokes,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    )
    v_lines_to_remove = cv2.bitwise_and(v_lines, cv2.bitwise_not(h_prot))

    # Combine lines to remove
    lines_to_remove = cv2.bitwise_or(h_lines_to_remove, v_lines_to_remove)

    # Subtract lines from binary image
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(lines_to_remove))

    # Remove isolated thin line slivers and speckles
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_cleaned = np.zeros_like(cleaned)

    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        carea = stats[i, cv2.CC_STAT_AREA]

        # Ignore microscopic speckles
        if carea < 12:
            continue
        # Discard flat isolated horizontal line slivers (e.g. width > 15 with height <= 2)
        if ch <= 2 and cw > 15:
            continue
        # Discard thin isolated vertical margin slivers
        if cw <= 2 and ch > 15:
            continue

        final_cleaned[labels == i] = 255

    return final_cleaned, lines_to_remove


def estimate_skew(binary, max_angle=25, step=0.5):
    """
    Estimate handwriting slant and tilt angle.
    Uses projection profile variance optimization.
    Returns angle in degrees (the angle to rotate the image by to make text horizontal).
    """
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 30:
        return 0.0

    h, w = binary.shape

    # Downscale for fast CPU projection search
    scale = 1.0
    if max(h, w) > 500:
        scale = 500.0 / max(h, w)
        small = cv2.resize(binary, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        small = binary

    sh, sw = small.shape
    center = (sw // 2, sh // 2)

    # Coarse search
    coarse_angles = np.arange(-max_angle, max_angle + 1.0, 1.5)
    best_score = -1.0
    best_coarse = 0.0

    for a in coarse_angles:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        proj = np.sum(rot > 0, axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_coarse = a

    # Fine search around best coarse angle
    fine_angles = np.arange(best_coarse - 2.0, best_coarse + 2.1, step)
    best_fine = best_coarse
    for a in fine_angles:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        proj = np.sum(rot > 0, axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_fine = a

    return float(best_fine)


def deskew_image(image, angle, border_value=0):
    """
    Rotate an image (binary or color) by a specific angle to level text lines.
    Uses BORDER_REPLICATE for color to prevent artificial border edges.
    """
    if abs(angle) < 0.2:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    is_color = len(image.shape) == 3
    flags = cv2.INTER_CUBIC if is_color else cv2.INTER_NEAREST
    border_mode = cv2.BORDER_REPLICATE if is_color else cv2.BORDER_CONSTANT

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=flags,
        borderMode=border_mode,
        borderValue=0 if not is_color else (255, 255, 255)
    )


def deskew(binary):
    """Backward-compatible deskew function."""
    angle = estimate_skew(binary)
    return deskew_image(binary, angle, border_value=0)


def remove_noise(binary):
    """Remove tiny connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 15:
            cleaned[labels == i] = 255

    return cleaned


def preprocess_image(image, return_color=False):
    """
    Comprehensive OpenCV preprocessing pipeline for handwritten notes:
    1. Illumination normalization & shadow removal (scale-adaptive background division)
    2. Bleed-through and reverse-side pen impression suppression
    3. Ruled notebook lines & margin lines removal from binary segmentation mask
    4. Preserves pristine, un-smudged color stream for high-accuracy TrOCR line crops
    5. Two-stage tilt & skew estimation via projection profile variance
    6. Document deskewing
    
    Args:
        image: BGR OpenCV image
        return_color: if True, returns (binary, clean_color, metadata)
                      if False, returns binary (backward-compatible)
    Returns:
        binary: white handwriting on black background (crisp, un-smudged, deskewed)
        cleaned_color: (optional) pristine color image (deskewed, un-smudged)
        metadata: (optional) dict with skew_angle, lines_removed_count, etc.
    """
    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = image.copy()
    else:
        gray = image.copy()
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Shadow removal and illumination normalization
    norm = remove_shadows(gray)

    # 3. Suppress bleed-through and binarize crisply
    binary, _ = suppress_bleed_through_and_binarize(norm)

    # 4. Remove ruled notebook lines from binary segmentation mask
    cleaned_binary, lines_mask = remove_ruled_lines(binary)

    # 5. Clean color stream: preserve natural RGB ink strokes without destructive inpainting!
    # TrOCR requires the sharp, un-smudged natural color image.
    cleaned_color = color.copy()

    # 6. Estimate skew and tilt on cleaned text
    skew_angle = estimate_skew(cleaned_binary)

    # 7. Deskew binary and color images
    if abs(skew_angle) >= 0.3:
        cleaned_binary = deskew_image(cleaned_binary, skew_angle, border_value=0)
        cleaned_color = deskew_image(cleaned_color, skew_angle, border_value=255)

    lines_removed_px = int(np.count_nonzero(lines_mask))
    metadata = {
        "skew_angle": skew_angle,
        "lines_removed_px": lines_removed_px,
        "has_ruled_lines": lines_removed_px > 80,
        "original_shape": image.shape,
        "processed_shape": cleaned_binary.shape
    }

    if return_color:
        return cleaned_binary, cleaned_color, metadata
    return cleaned_binary