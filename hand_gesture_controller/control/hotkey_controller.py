from __future__ import annotations

"""Listens for global hotkeys to toggle controls, switch modes, and request exit.
Uses pynput if available.
"""

from threading import Lock

try:
	from pynput import keyboard
except Exception:
	keyboard = None


class HotkeyController:
	def __init__(self, enabled: bool = True) -> None:
		self._enabled = enabled and keyboard is not None
		self._controls_enabled = True
		self._exit_requested = False
		self._mode_toggle_requested = False
		self._listener = None
		self._lock = Lock()

	def _on_press(self, key) -> None:
		if keyboard is None:
			return
		with self._lock:
			if key == keyboard.Key.f8:
				self._controls_enabled = not self._controls_enabled
			elif key == keyboard.Key.f9:
				self._mode_toggle_requested = True
			elif key == keyboard.Key.f10:
				self._exit_requested = True

	def start(self) -> None:
		if not self._enabled or keyboard is None:
			return
		self._listener = keyboard.Listener(on_press=self._on_press)
		self._listener.start()

	def stop(self) -> None:
		if self._listener is not None:
			self._listener.stop()
			self._listener = None

	def controls_enabled(self) -> bool:
		with self._lock:
			return self._controls_enabled

	def should_exit(self) -> bool:
		with self._lock:
			return self._exit_requested

	def consume_mode_toggle(self) -> bool:
		with self._lock:
			requested = self._mode_toggle_requested
			self._mode_toggle_requested = False
			return requested

	def available(self) -> bool:
		return self._enabled
