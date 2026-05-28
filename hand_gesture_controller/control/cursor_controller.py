from __future__ import annotations

"""Sends OS-level cursor and input events using xdotool, pynput, or pyautogui.
Provides click, scroll, tab, and zoom actions for gesture control.
"""

import os
import shutil
import subprocess
import time

try:
	from pynput import keyboard as pynput_keyboard
	from pynput import mouse as pynput_mouse
except Exception:
	pynput_keyboard = None
	pynput_mouse = None

try:
	import pyautogui
except Exception:
	pyautogui = None


class _PyAutoGuiBackend:
	name = "pyautogui"

	def __init__(self) -> None:
		if pyautogui is None:
			raise RuntimeError("pyautogui not available")
		pyautogui.PAUSE = 0
		pyautogui.FAILSAFE = False

	def screen_size(self) -> tuple[int, int]:
		width, height = pyautogui.size()
		return int(width), int(height)

	def move_to(self, x: int, y: int) -> None:
		pyautogui.moveTo(x, y)

	def click_left(self) -> None:
		pyautogui.click(button="left")

	def click_right(self) -> None:
		pyautogui.click(button="right")

	def scroll(self, amount: int) -> None:
		pyautogui.scroll(amount)

	def next_tab(self) -> None:
		pyautogui.hotkey("ctrl", "tab")

	def prev_tab(self) -> None:
		pyautogui.hotkey("ctrl", "shift", "tab")

	def open_tab(self) -> None:
		pyautogui.hotkey("ctrl", "t")

	def close_tab(self) -> None:
		pyautogui.hotkey("ctrl", "w")

	def zoom_in(self) -> None:
		try:
			pyautogui.hotkey("ctrl", "plus")
		except Exception:
			pyautogui.hotkey("ctrl", "equal")

	def zoom_out(self) -> None:
		pyautogui.hotkey("ctrl", "minus")


class _XDoToolBackend:
	name = "xdotool"

	def __init__(self) -> None:
		if shutil.which("xdotool") is None:
			raise RuntimeError("xdotool not installed")
		if not os.environ.get("DISPLAY"):
			raise RuntimeError("DISPLAY is not set")
		self._screen_cache: tuple[int, int] | None = None
		self._screen_cache_until_ms = 0.0

	@staticmethod
	def _run(*args: str) -> str:
		result = subprocess.run(
			["xdotool", *args],
			check=False,
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			raise RuntimeError((result.stderr or result.stdout or "xdotool command failed").strip())
		return result.stdout.strip()

	def screen_size(self) -> tuple[int, int]:
		now_ms = time.monotonic() * 1000.0
		if self._screen_cache and now_ms < self._screen_cache_until_ms:
			return self._screen_cache

		output = self._run("getdisplaygeometry")
		parts = output.split()
		if len(parts) < 2:
			raise RuntimeError(f"unexpected geometry output: {output}")

		width = int(parts[0])
		height = int(parts[1])
		self._screen_cache = (width, height)
		self._screen_cache_until_ms = now_ms + 2000.0
		return self._screen_cache

	def move_to(self, x: int, y: int) -> None:
		self._run("mousemove", str(x), str(y))

	def click_left(self) -> None:
		self._run("click", "1")

	def click_right(self) -> None:
		self._run("click", "3")

	def scroll(self, amount: int) -> None:
		button = "4" if amount > 0 else "5"
		steps = max(1, min(10, abs(amount) // 40 or 1))
		self._run("click", "--repeat", str(steps), button)

	def next_tab(self) -> None:
		self._run("key", "ctrl+Tab")

	def prev_tab(self) -> None:
		self._run("key", "ctrl+shift+Tab")

	def open_tab(self) -> None:
		self._run("key", "ctrl+t")

	def close_tab(self) -> None:
		self._run("key", "ctrl+w")

	def zoom_in(self) -> None:
		self._run("key", "ctrl+plus")

	def zoom_out(self) -> None:
		self._run("key", "ctrl+minus")


class _PynputBackend:
	name = "pynput"

	def __init__(self) -> None:
		if pynput_mouse is None or pynput_keyboard is None:
			raise RuntimeError("pynput mouse/keyboard not available")
		self._mouse = pynput_mouse.Controller()
		self._keyboard = pynput_keyboard.Controller()
		self._screen_size = self._detect_screen_size()

	@staticmethod
	def _detect_screen_size() -> tuple[int, int]:
		try:
			import tkinter as tk
		except Exception:
			return (1920, 1080)

		root = tk.Tk()
		root.withdraw()
		try:
			return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
		finally:
			root.destroy()

	def _hotkey(self, *keys) -> None:
		for key in keys:
			self._keyboard.press(key)
		for key in reversed(keys):
			self._keyboard.release(key)

	def screen_size(self) -> tuple[int, int]:
		return self._screen_size

	def move_to(self, x: int, y: int) -> None:
		self._mouse.position = (x, y)

	def click_left(self) -> None:
		self._mouse.click(pynput_mouse.Button.left)

	def click_right(self) -> None:
		self._mouse.click(pynput_mouse.Button.right)

	def scroll(self, amount: int) -> None:
		steps = int(amount / 40)
		if steps == 0 and amount != 0:
			steps = 1 if amount > 0 else -1
		self._mouse.scroll(0, steps)

	def next_tab(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, pynput_keyboard.Key.tab)

	def prev_tab(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, pynput_keyboard.Key.shift, pynput_keyboard.Key.tab)

	def open_tab(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, "t")

	def close_tab(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, "w")

	def zoom_in(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, "+")

	def zoom_out(self) -> None:
		self._hotkey(pynput_keyboard.Key.ctrl, "-")


class CursorController:
	def __init__(
		self,
		smoothing_alpha: float,
		move_interval_ms: int = 10,
		min_move_pixels: float = 1.5,
	) -> None:
		self.smoothing_alpha = smoothing_alpha
		self.move_interval_ms = move_interval_ms
		self.min_move_pixels = min_move_pixels
		self._smoothed_x: float | None = None
		self._smoothed_y: float | None = None
		self._last_move_ms = 0.0
		self._last_sent_x: float | None = None
		self._last_sent_y: float | None = None
		self._backend = self._select_backend()
		self.backend_name = self._backend.name if self._backend is not None else "disabled"

	def _select_backend(self):
		preferred = os.environ.get("HGC_INPUT_BACKEND", "auto").strip().lower()
		import sys

		def _build(name: str):
			if name == "pyautogui":
				return _PyAutoGuiBackend()
			if name == "pynput":
				return _PynputBackend()
			if name == "xdotool":
				return _XDoToolBackend()
			raise RuntimeError(f"unsupported backend '{name}'")

		if preferred != "auto":
			try:
				backend = _build(preferred)
				msg = f"[CursorController] Using preferred backend: {preferred}\n"
				sys.stderr.write(msg)
				sys.stderr.flush()
				return backend
			except Exception as e:
				msg = f"[CursorController] Failed to initialize preferred backend '{preferred}': {e}\n"
				msg += f"[CursorController] DISPLAY={os.environ.get('DISPLAY', 'NOT SET')}, xdotool={shutil.which('xdotool')}\n"
				sys.stderr.write(msg)
				sys.stderr.flush()
				return None

		for candidate in ("pyautogui", "pynput", "xdotool"):
			try:
				backend = _build(candidate)
				msg = f"[CursorController] Auto-selected backend: {candidate}\n"
				sys.stderr.write(msg)
				sys.stderr.flush()
				return backend
			except Exception as e:
				msg = f"[CursorController] Failed to initialize candidate '{candidate}': {e}\n"
				sys.stderr.write(msg)
				sys.stderr.flush()
				continue
		msg = "[CursorController] No backend available - cursor control disabled\n"
		sys.stderr.write(msg)
		sys.stderr.flush()
		return None

	def _smooth(self, old_value: float | None, new_value: float) -> float:
		if old_value is None:
			return new_value
		alpha = self.smoothing_alpha
		return old_value * (1.0 - alpha) + new_value * alpha

	def _call(self, fn) -> bool:
		if self._backend is None:
			return False
		try:
			fn()
			return True
		except Exception:
			self._backend = None
			self.backend_name = "disabled"
			return False

	def move_from_normalized(self, x: float, y: float) -> bool:
		if self._backend is None:
			return False

		now_ms = time.monotonic() * 1000.0
		if (now_ms - self._last_move_ms) < self.move_interval_ms:
			return False

		try:
			screen_w, screen_h = self._backend.screen_size()
		except Exception:
			self._backend = None
			self.backend_name = "disabled"
			return False
		target_x = x * screen_w
		target_y = y * screen_h

		self._smoothed_x = self._smooth(self._smoothed_x, target_x)
		self._smoothed_y = self._smooth(self._smoothed_y, target_y)

		if self._last_sent_x is not None and self._last_sent_y is not None:
			dx = abs(self._smoothed_x - self._last_sent_x)
			dy = abs(self._smoothed_y - self._last_sent_y)
			if dx < self.min_move_pixels and dy < self.min_move_pixels:
				return False

		if not self._call(lambda: self._backend.move_to(int(self._smoothed_x), int(self._smoothed_y))):
			return False
		self._last_move_ms = now_ms
		self._last_sent_x = self._smoothed_x
		self._last_sent_y = self._smoothed_y
		return True

	def click_left(self) -> bool:
		return self._call(lambda: self._backend.click_left())

	def click_right(self) -> bool:
		return self._call(lambda: self._backend.click_right())

	def scroll(self, amount: int) -> bool:
		return self._call(lambda: self._backend.scroll(amount))

	def next_tab(self) -> bool:
		return self._call(lambda: self._backend.next_tab())

	def prev_tab(self) -> bool:
		return self._call(lambda: self._backend.prev_tab())

	def open_tab(self) -> bool:
		return self._call(lambda: self._backend.open_tab())

	def close_tab(self) -> bool:
		return self._call(lambda: self._backend.close_tab())

	def zoom_in(self) -> bool:
		return self._call(lambda: self._backend.zoom_in())

	def zoom_out(self) -> bool:
		return self._call(lambda: self._backend.zoom_out())
