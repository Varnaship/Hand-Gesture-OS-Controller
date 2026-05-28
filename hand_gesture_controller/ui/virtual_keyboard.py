from __future__ import annotations

"""Simple on-screen virtual keyboard driven by index-finger tracking.

Usage:
- Instantiate `VirtualKeyboard()` and call `update(hands)` every frame from main loop.
- The keyboard will draw a window and accept clicks when an index finger is bent
  (tip near pip) while hovering over a key.
"""

import time
from typing import List

import cv2

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

from hand_gesture_controller.tracking.hand_tracker import TrackedHand


class VirtualKeyboard:
    def __init__(self, window_name: str = "Virtual Keyboard") -> None:
        self.window_name = window_name
        self.width = 900
        self.height = 300
        # simple QWERTY rows
        self.rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM"),
            ["SPACE"],
        ]
        self.key_w = 70
        self.key_h = 60
        self.margin_x = 20
        self.margin_y = 20
        self.last_click_ms: dict[str, float] = {}
        self.click_cooldown = 350
        cv2.namedWindow(self.window_name)
        self.open = True

    def _key_at_pos(self, x: int, y: int) -> str | None:
        # compute bounding boxes for keys and return key label under point
        y_offset = self.margin_y
        for r_i, row in enumerate(self.rows):
            # center row horizontally
            row_width = len(row) * (self.key_w + 8)
            x_start = (self.width - row_width) // 2
            x_cursor = x_start
            for k in row:
                x1 = x_cursor
                y1 = y_offset
                x2 = x_cursor + self.key_w
                y2 = y_offset + self.key_h
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return k
                x_cursor += self.key_w + 8
            y_offset += self.key_h + 12
        return None

    def _send_key(self, key: str) -> None:
        try:
            if key == "SPACE":
                k = "space"
            else:
                k = key.lower()
            if pyautogui is not None:
                pyautogui.press(k)
                return
            if pynput_keyboard is not None:
                kb = pynput_keyboard.Controller()
                if k == "space":
                    kb.press(pynput_keyboard.Key.space)
                    kb.release(pynput_keyboard.Key.space)
                else:
                    kb.press(k)
                    kb.release(k)
                return
        except Exception:
            pass

    def _finger_bent(self, hand: TrackedHand) -> bool:
        # index tip (8) vs pip (6)
        ax, ay, _ = hand.landmarks_norm[8]
        bx, by, _ = hand.landmarks_norm[6]
        # normalized distance
        d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return d < 0.03

    def update(self, hands: List[TrackedHand]) -> None:
        if not self.open:
            return
        img = 255 * (1 - 0) * (1,)  # placeholder to satisfy array creation
        img = 255 * (1,)  # suppressed - create a real image next
        import numpy as np

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # draw keys
        y_offset = self.margin_y
        for r_i, row in enumerate(self.rows):
            row_width = len(row) * (self.key_w + 8)
            x_start = (self.width - row_width) // 2
            x_cursor = x_start
            for k in row:
                x1 = x_cursor
                y1 = y_offset
                x2 = x_cursor + self.key_w
                y2 = y_offset + self.key_h
                cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(img, k, (x1 + 10, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                x_cursor += self.key_w + 8
            y_offset += self.key_h + 12

        now_ms = time.monotonic() * 1000.0
        # draw finger cursors and handle clicks
        for hand in hands:
            if not hand.landmarks_norm:
                continue
            ix, iy, _ = hand.landmarks_norm[8]
            px = int(ix * self.width)
            py = int(iy * self.height)
            cv2.circle(img, (px, py), 10, (0, 0, 255), -1)
            # detect bent
            if self._finger_bent(hand):
                key = self._key_at_pos(px, py)
                if key:
                    last = self.last_click_ms.get(key, 0.0)
                    if (now_ms - last) >= self.click_cooldown:
                        self._send_key(key)
                        self.last_click_ms[key] = now_ms
                        # flash key
                        x_offset = self.margin_x
        # show window
        cv2.imshow(self.window_name, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            self.open = False
            cv2.destroyWindow(self.window_name)

    def close(self) -> None:
        if self.open:
            self.open = False
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
