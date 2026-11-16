from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class LineRegion:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    ink_width: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    indices = np.arange(256, dtype=np.float64)
    sum_total = float((indices * histogram).sum())
    weight_background = 0.0
    sum_background = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _select_gap_boundaries(projection: np.ndarray, height: int) -> tuple[int, int, list[int]]:
    smooth = np.convolve(projection, np.ones(3) / 3.0, mode="same")
    content_rows = np.where(smooth > max(0.003, float(np.percentile(smooth, 15))))[0]
    if not len(content_rows):
        return 0, height, []
    content_start = int(content_rows[0])
    content_end = int(content_rows[-1]) + 1
    interior = smooth[content_start:content_end]
    gap_threshold = min(0.08, max(0.012, float(np.percentile(interior, 30)) + 0.008))
    low = interior < gap_threshold
    minimum_gap = max(3, int(round(height * 0.012)))
    candidates: list[tuple[float, int]] = []
    start: int | None = None
    for index, value in enumerate(low):
        if value and start is None:
            start = index
        if (not value or index == len(low) - 1) and start is not None:
            end = index if not value else index + 1
            if end - start >= minimum_gap:
                absolute_start = content_start + start
                absolute_end = content_start + end
                center = (absolute_start + absolute_end) // 2
                mean_density = float(smooth[absolute_start:absolute_end].mean())
                score = mean_density - (absolute_end - absolute_start) / max(height, 1) * 0.05
                candidates.append((score, center))
            start = None

    minimum_line_height = max(18, int(round(height * 0.105)))
    selected = [content_start, content_end]
    for _, center in sorted(candidates):
        if all(abs(center - boundary) >= minimum_line_height for boundary in selected):
            selected.append(center)
        if len(selected) >= 7:
            break
    return content_start, content_end, sorted(selected[2:])


def detect_line_regions(image: Image.Image, padding_ratio: float = 0.018) -> list[LineRegion]:
    gray_image = ImageOps.autocontrast(ImageOps.grayscale(image))
    gray = np.asarray(gray_image, dtype=np.uint8)
    height, width = gray.shape
    threshold = min(230, _otsu_threshold(gray) + 20)
    ink = gray < threshold
    projection = ink.mean(axis=1)
    content_start, content_end, gap_boundaries = _select_gap_boundaries(projection, height)
    boundaries = [content_start] + gap_boundaries + [content_end]

    pad_y = max(2, int(round(height * padding_ratio)))
    pad_x = max(4, int(round(width * padding_ratio)))
    regions: list[LineRegion] = []
    for top, bottom in zip(boundaries, boundaries[1:]):
        top = max(0, int(top) - pad_y)
        bottom = min(height, int(bottom) + pad_y)
        band = ink[top:bottom]
        columns = np.where(band.any(axis=0))[0]
        if len(columns):
            x1 = max(0, int(columns[0]) - pad_x)
            x2 = min(width, int(columns[-1]) + 1 + pad_x)
        else:
            x1, x2 = 0, width
        if bottom - top < max(12, int(height * 0.05)):
            continue
        regions.append(LineRegion(len(regions), x1, top, x2, bottom, max(1, x2 - x1)))
    return regions or [LineRegion(0, 0, 0, width, height, width)]


def split_target_across_lines(text: str, line_weights: list[float]) -> list[str]:
    words = text.split()
    line_count = min(len(line_weights), len(words))
    if line_count <= 1:
        return [" ".join(words)]
    weights = np.asarray(line_weights[:line_count], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0)
    total_characters = sum(len(word) for word in words) + max(0, len(words) - 1)
    prefix = [0]
    for index, word in enumerate(words):
        prefix.append(prefix[-1] + len(word) + (1 if index else 0))

    infinity = float("inf")
    cost = [[infinity] * (len(words) + 1) for _ in range(line_count + 1)]
    back = [[-1] * (len(words) + 1) for _ in range(line_count + 1)]
    cost[0][0] = 0.0
    for line_index in range(1, line_count + 1):
        for end in range(line_index, len(words) + 1):
            for start in range(line_index - 1, end):
                if cost[line_index - 1][start] == infinity:
                    continue
                segment_length = prefix[end] - prefix[start] - (1 if start > 0 else 0)
                observed_ratio = segment_length / max(total_characters, 1)
                penalty = (observed_ratio - weights[line_index - 1]) ** 2
                candidate = cost[line_index - 1][start] + penalty
                if candidate < cost[line_index][end]:
                    cost[line_index][end] = candidate
                    back[line_index][end] = start

    boundaries = [len(words)]
    end = len(words)
    for line_index in range(line_count, 0, -1):
        end = back[line_index][end]
        boundaries.append(end)
    boundaries.reverse()
    return [" ".join(words[start:end]) for start, end in zip(boundaries, boundaries[1:])]
