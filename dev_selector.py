#!/usr/bin/env python3
"""Interactive magic-wand-style selector for stomatal training examples.

Usage: python dev_selector.py --image Stomata_Images/IMG_3018.JPG

Controls: click=select region, +/- or mouse wheel=zoom, arrows/HJKL=pan,
[/]=change tolerance, s=save, c=clear, q=quit.
Selections are saved as PNG masks and JSON metadata under dev_annotations/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class Selector:
    def __init__(self, image_path: Path, output: Path, max_width: int = 1800):
        self.image_path = image_path
        self.output = output
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        self.original_shape = image.shape[:2]
        self.image = image
        self.scale = min(1.0, max_width / image.shape[1])
        self.display_size = (
            max(1, round(image.shape[1] * self.scale)),
            max(1, round(image.shape[0] * self.scale)),
        )
        self.lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.selection = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.tolerance = 6
        self.zoom = 1.0
        self.cursor = (self.display_size[0] // 2, self.display_size[1] // 2)
        self.view_center = [self.image.shape[1] / 2, self.image.shape[0] / 2]
        self.window = "Stomata dev selector"

    def click(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            self.cursor = (x, y)
            self.zoom_at(x, y, 1.25 if flags > 0 else 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = self.to_base(x, y)
        mask = np.zeros((self.lab.shape[0] + 2, self.lab.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(self.lab.copy(), mask, (x, y), (0, 0, 0),
                      (self.tolerance,) * 3, (self.tolerance,) * 3,
                      flags=cv2.FLOODFILL_MASK_ONLY | (4 << 8))
        region = (mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
        if int(np.count_nonzero(region)) > 0.25 * region.size:
            print("Selection rejected: region is larger than 25% of the image. Lower tolerance.")
            return
        self.selection = cv2.bitwise_or(self.selection, region)

    def viewport(self):
        h, w = self.image.shape[:2]
        crop_w = max(1, round(w / self.zoom))
        crop_h = max(1, round(h / self.zoom))
        x0 = round(self.view_center[0] - crop_w / 2)
        y0 = round(self.view_center[1] - crop_h / 2)
        x0 = int(np.clip(x0, 0, max(0, w - crop_w)))
        y0 = int(np.clip(y0, 0, max(0, h - crop_h)))
        self.view_center = [x0 + crop_w / 2, y0 + crop_h / 2]
        return x0, y0, crop_w, crop_h

    def render(self):
        x0, y0, crop_w, crop_h = self.viewport()
        view = self.image[y0:y0 + crop_h, x0:x0 + crop_w].copy()
        selected = self.selection[y0:y0 + crop_h, x0:x0 + crop_w] > 0
        view[selected] = (180, 0, 220)
        view = cv2.resize(
            view,
            self.display_size,
            interpolation=cv2.INTER_AREA
            if crop_w > self.display_size[0] else cv2.INTER_LINEAR,
        )
        cv2.putText(view, f"zoom={self.zoom:.2f}x tol={self.tolerance} | +/- or wheel zoom | arrows/HJKL pan | [/] tolerance | click select | s save | c clear | q quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return view

    def to_base(self, x, y):
        h, w = self.image.shape[:2]
        display_w, display_h = self.display_size
        x0, y0, crop_w, crop_h = self.viewport()
        return (int(np.clip(x0 + x * crop_w / display_w, 0, w - 1)),
                int(np.clip(y0 + y * crop_h / display_h, 0, h - 1)))

    def zoom_at(self, x, y, factor):
        old_x, old_y = self.to_base(x, y)
        old_zoom = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 12.0))
        if self.zoom == old_zoom:
            return
        h, w = self.image.shape[:2]
        display_w, display_h = self.display_size
        crop_w, crop_h = w / self.zoom, h / self.zoom
        self.view_center[0] = old_x + crop_w * (0.5 - x / display_w)
        self.view_center[1] = old_y + crop_h * (0.5 - y / display_h)
        self.viewport()

    def pan(self, dx, dy):
        _, _, crop_w, crop_h = self.viewport()
        self.view_center[0] += dx * crop_w
        self.view_center[1] += dy * crop_h
        self.viewport()

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        existing = sorted(self.output.glob(f"{stem}_mask_*.png"))
        index = len(existing) + 1
        mask_path = self.output / f"{stem}_mask_{index:03d}.png"
        cv2.imwrite(str(mask_path), self.selection)
        metadata = {"image": self.image_path.name, "mask": mask_path.name,
                    "tolerance": self.tolerance, "preview_scale": self.scale,
                    "original_shape": list(self.original_shape)}
        with (self.output / f"{stem}_mask_{index:03d}.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Saved {mask_path}")

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, *self.display_size)
        cv2.setMouseCallback(self.window, self.click)
        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and self.selection.any():
                self.save()
            elif key == ord("c"):
                self.selection.fill(0)
            elif key in (ord("+"), ord("=")):
                self.zoom_at(*self.cursor, 1.25)
            elif key in (ord("-"), 45):
                self.zoom_at(*self.cursor, 0.8)
            elif key == ord("]"):
                self.tolerance = min(100, self.tolerance + 1)
            elif key == ord("["):
                self.tolerance = max(1, self.tolerance - 1)
            elif key in (ord("h"), 81):
                self.pan(-0.15, 0)
            elif key in (ord("l"), 83):
                self.pan(0.15, 0)
            elif key in (ord("k"), 82):
                self.pan(0, -0.15)
            elif key in (ord("j"), 84):
                self.pan(0, 0.15)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("dev_annotations"))
    args = parser.parse_args()
    Selector(args.image, args.output).run()


if __name__ == "__main__":
    main()
