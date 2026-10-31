import cv2
import numpy as np


def remove_small_noise(binary):
    """Remove tiny noise components and specks."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 15:
            cleaned[labels == i] = 255

    return cleaned


def detect_lines(binary):
    """
    Advanced text line detection that handles tilted, uneven handwriting
    without merging distinct lines or skipping short lines.
    
    Returns:
        List of line bounding tuples: [(y1, y2, x1, x2), ...]
    """
    h_img, w_img = binary.shape

    # 1. Analyze connected components to estimate character height and stroke scale
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid_heights = []
    valid_widths = []
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        carea = stats[i, cv2.CC_STAT_AREA]
        if ch >= 8 and cw >= 4 and carea >= 18:
            valid_heights.append(ch)
            valid_widths.append(cw)

    median_h = int(np.median(valid_heights)) if valid_heights else 26

    # 2. Adaptive horizontal dilation kernel tailored to median character size
    # Kernel width is scaled to bridge word gaps on the same line
    kernel_w = max(22, int(median_h * 2.4))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 2))
    dilated = cv2.dilate(binary, h_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise and full-frame artifacts
        if h < 10 or w < 16:
            continue
        if h > h_img * 0.92 and w > w_img * 0.92:
            continue

        # Ensure box contains sufficient actual text stroke pixels
        sub_bin = binary[y:y + h, x:x + w]
        ink_pixels = np.count_nonzero(sub_bin)
        if ink_pixels < 140:
            continue

        # Ensure box contains genuine character connected components (rejects shadow smudges and dust)
        num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(sub_bin, connectivity=8)
        valid_cc = sum(1 for i in range(1, num_cc) if cc_stats[i, cv2.CC_STAT_AREA] >= 12 and cc_stats[i, cv2.CC_STAT_HEIGHT] >= 5)
        if valid_cc < 3:
            continue

        candidate_boxes.append([y, y + h, x, x + w])

    # Fallback to horizontal projection profile if dilation finds no valid contour
    if not candidate_boxes:
        proj = np.sum(binary > 0, axis=1)
        in_line = False
        start = 0
        thresh = max(3, int(w_img * 0.005))
        for y, val in enumerate(proj):
            if val > thresh and not in_line:
                start = y
                in_line = True
            elif val <= thresh and in_line:
                if y - start >= 8:
                    candidate_boxes.append([start, y, 0, w_img])
                in_line = False
        if in_line and h_img - start >= 8:
            candidate_boxes.append([start, h_img, 0, w_img])

    # 3. Valley splitting for tall boxes that accidentally merged two adjacent lines
    split_boxes = []
    for box in candidate_boxes:
        y1, y2, x1, x2 = box
        bh = y2 - y1
        if bh > median_h * 2.6 and bh >= 50:
            sub_bin = binary[y1:y2, x1:x2]
            row_sums = np.sum(sub_bin > 0, axis=1).astype(np.float32)

            # Gaussian smoothing on 1D projection profile
            k_smooth = max(5, int(median_h * 0.4))
            if k_smooth % 2 == 0:
                k_smooth += 1
            kernel_1d = cv2.getGaussianKernel(k_smooth, -1)
            smoothed = cv2.filter2D(row_sums, -1, kernel_1d).flatten()

            # Find local peaks
            peaks = []
            for k in range(1, len(smoothed) - 1):
                if smoothed[k] > smoothed[k - 1] and smoothed[k] >= smoothed[k + 1]:
                    if smoothed[k] > 0.28 * np.max(smoothed):
                        peaks.append(k)

            # If two substantial peaks exist with enough vertical distance, split at deepest valley
            if len(peaks) >= 2 and (peaks[-1] - peaks[0]) >= median_h * 0.8:
                valley_idx = peaks[0] + int(np.argmin(smoothed[peaks[0]:peaks[-1]]))
                if smoothed[valley_idx] < 0.40 * min(smoothed[peaks[0]], smoothed[peaks[-1]]):
                    total_ink = np.sum(row_sums)
                    top_ink = np.sum(row_sums[:valley_idx])
                    bot_ink = np.sum(row_sums[valley_idx:])
                    # Both split halves must contain at least 20% of the ink mass (prevents descender splitting)
                    if top_ink >= 0.20 * total_ink and bot_ink >= 0.20 * total_ink:
                        split_boxes.append([y1, y1 + valley_idx, x1, x2])
                        split_boxes.append([y1 + valley_idx, y2, x1, x2])
                        continue

        split_boxes.append(box)

    # 4. Group horizontally separated fragments of the SAME text line
    # (Without merging vertically distinct lines)
    split_boxes.sort(key=lambda b: (b[0] + b[1]) / 2.0)
    line_levels = []
    for box in split_boxes:
        y1, y2, x1, x2 = box
        cy = (y1 + y2) / 2.0
        bh = y2 - y1

        matched = False
        for lvl in line_levels:
            lvl_y1, lvl_y2, lvl_x1, lvl_x2 = lvl
            lvl_cy = (lvl_y1 + lvl_y2) / 2.0
            lvl_h = lvl_y2 - lvl_y1

            # Only merge if on the same horizontal text level
            vert_dist = abs(cy - lvl_cy)
            max_h = max(bh, lvl_h)
            if vert_dist < max_h * 0.42:
                lvl[0] = min(lvl_y1, y1)
                lvl[1] = max(lvl_y2, y2)
                lvl[2] = min(lvl_x1, x1)
                lvl[3] = max(lvl_x2, x2)
                matched = True
                break
        if not matched:
            line_levels.append([y1, y2, x1, x2])

    # Sort lines strictly from top to bottom
    line_levels.sort(key=lambda b: (b[0] + b[1]) / 2.0)
    return [tuple(b) for b in line_levels]


def get_line_crops(image, line_regions, pad_y=None, pad_x=None):
    """
    Safely extract padded line crop images from the source image.
    Uses adaptive padding so ascenders and descenders are fully captured,
    while bounding by neighboring lines to prevent cross-line bleed.
    Returns:
        List of (crop_img, (y1_crop, y2_crop, x1_crop, x2_crop))
    """
    h_img, w_img = image.shape[:2]
    crops = []
    num_lines = len(line_regions)

    for i, (y1, y2, x1, x2) in enumerate(line_regions):
        lh = y2 - y1
        py = pad_y if pad_y is not None else max(8, int(lh * 0.18))
        px = pad_x if pad_x is not None else max(10, int(lh * 0.22))

        # Prevent bleeding into neighboring lines
        min_y = 0 if i == 0 else (line_regions[i - 1][1] + y1) // 2
        max_y = h_img if i == num_lines - 1 else (y2 + line_regions[i + 1][0]) // 2

        y1_c = max(min_y, y1 - py)
        y2_c = min(max_y, y2 + py)
        x1_c = max(0, x1 - px)
        x2_c = min(w_img, x2 + px)

        if y2_c <= y1_c or x2_c <= x1_c:
            continue

        crop = image[y1_c:y2_c, x1_c:x2_c]
        if crop.size > 0:
            crops.append((crop, (y1_c, y2_c, x1_c, x2_c)))

    return crops


def split_wide_component(component):
    """
    Attempt to split connected components that are too wide.
    This helps when multiple handwritten characters touch.
    """
    h, w = component.shape
    projection = np.sum(component > 0, axis=0)

    splits = []
    in_char = False
    start = 0

    for x, value in enumerate(projection):
        if value > 0 and not in_char:
            start = x
            in_char = True
        elif value == 0 and in_char:
            end = x
            if end - start > 2:
                splits.append((start, end))
            in_char = False

    if in_char:
        splits.append((start, w))

    return splits


def prepare_character(char_img):
    """
    Prepare one character for EMNIST CNN.
    Output: 28x28 normalized float32 character image.
    """
    h, w = char_img.shape
    size = max(h, w)
    canvas = np.zeros((size + 10, size + 10), dtype=np.uint8)

    y = (canvas.shape[0] - h) // 2
    x = (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = char_img

    resized = cv2.resize(
        canvas,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    resized = resized.astype("float32") / 255.0
    return resized


def merge_broken_parts(line_boxes):
    """
    Merge connected components that are likely parts of the same letter.
    Handles broken letters like 'i', 'j', 't' where the dot/crossbar
    is a separate connected component from the stem.
    """
    if len(line_boxes) <= 1:
        return line_boxes

    merged = []
    used = [False] * len(line_boxes)

    for i, box1 in enumerate(line_boxes):
        if used[i]:
            continue

        used[i] = True
        x1, y1, w1, h1 = box1
        cx1 = x1 + w1 / 2.0
        merged_box = [x1, y1, w1, h1]

        for j in range(i + 1, len(line_boxes)):
            if used[j]:
                continue

            x2, y2, w2, h2 = line_boxes[j]
            cx2 = x2 + w2 / 2.0

            horiz_dist = abs(cx1 - cx2)
            max_w = max(w1, w2)

            y_gap = max(
                0,
                max(merged_box[1], y2)
                - min(merged_box[1] + merged_box[3], y2 + h2)
            )

            # Merge if horizontally aligned and vertically close
            if horiz_dist < max_w / 2.0 + 4 and y_gap < 12:
                new_x = min(merged_box[0], x2)
                new_y = min(merged_box[1], y2)
                new_w = max(merged_box[0] + merged_box[2], x2 + w2) - new_x
                new_h = max(merged_box[1] + merged_box[3], y2 + h2) - new_y

                if new_w <= 65 and new_h <= 65:
                    merged_box = [new_x, new_y, new_w, new_h]
                    used[j] = True

        merged.append(tuple(merged_box))

    return merged


def segment_characters(binary):
    """
    Main character segmentation function for EMNIST isolated character classification.

    Returns:
        characters -> processed 28x28 character images (float32, 0..1)
        boxes -> bounding boxes (x, y, w, h) in page coordinates
        line_ids -> line number for each character
        line_regions -> (y1, y2, x1, x2) for all detected text lines
    """
    # Make sure handwriting is white on black background
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Remove tiny noise
    binary = remove_small_noise(binary)

    # Detect text line regions
    line_regions = detect_lines(binary)

    characters = []
    boxes = []
    line_ids = []

    for line_id, (y1, y2, lx1, lx2) in enumerate(line_regions):
        line_img = binary[y1:y2, :]

        contours, _ = cv2.findContours(
            line_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        line_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 2 or h < 4:
                continue
            line_boxes.append((x, y, w, h))

        # Sort left -> right
        line_boxes.sort(key=lambda b: b[0])

        # Merge broken letter parts (i-dots, t-crossbars, etc.)
        line_boxes = merge_broken_parts(line_boxes)

        # Split wide components (touching letters)
        final_boxes = []
        for x, y, w, h in line_boxes:
            if w > 20:
                char_img = line_img[y:y + h, x:x + w]
                splits = split_wide_component(char_img)
                for sx, sw in splits:
                    if sw >= 3:
                        final_boxes.append((x + sx, y, sw, h))
            else:
                final_boxes.append((x, y, w, h))

        # Sort left -> right
        final_boxes.sort(key=lambda b: b[0])

        for x, y, w, h in final_boxes:
            char_img = line_img[y:y + h, x:x + w]
            prepared = prepare_character(char_img)
            characters.append(prepared)
            boxes.append((x, y + y1, w, h))
            line_ids.append(line_id)

    return characters, boxes, line_ids, line_regions


def draw_boxes(image, boxes, color=(255, 51, 0), thickness=2):
    """Draw character bounding boxes on image."""
    output = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )
    return output