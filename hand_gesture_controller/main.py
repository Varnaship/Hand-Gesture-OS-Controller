from __future__ import annotations

"""Application entry point for the Hand Gesture Controller.
Initializes camera, hand tracking, gesture detection, UI overlay, and action dispatch.
Run via start_app.sh for preview mode or start_control.sh for headless service mode.
"""

import argparse
from collections import deque
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from hand_gesture_controller.camera.camera_stream import CameraStream
from hand_gesture_controller.config import AppConfig
from hand_gesture_controller.control.action_dispatcher import ActionDispatcher
from hand_gesture_controller.control.cursor_controller import CursorController
from hand_gesture_controller.control.hotkey_controller import HotkeyController
from hand_gesture_controller.gesture.gesture_engine import GestureEngine
from hand_gesture_controller.tracking.hand_tracker import HandTracker
from hand_gesture_controller.ui.overlay import Overlay
from hand_gesture_controller.ui.virtual_keyboard import VirtualKeyboard
from pathlib import PurePosixPath
import webbrowser


# Absolute path to project directory for consistent file lookups regardless of cwd
_PROJECT_DIR = Path(__file__).parent.parent.resolve()

REAL_BTN = (10, 10, 400, 55)
CTRL_BTN = (10, 65, 400, 110)
STOP_BTN = (10, 120, 400, 165)
SETTINGS_BTN = (10, 170, 400, 215)
PANEL_NAME = "Gesture Controls"
CONTROL_PID_FILE = _PROJECT_DIR / ".control.pid"
START_CONTROL_SCRIPT = _PROJECT_DIR / "start_control.sh"
APP_LOCK_FILE = _PROJECT_DIR / ".app.lock"


def _pointer_active(hand) -> bool:
	def _dist(a_idx: int, b_idx: int) -> float:
		ax, ay, _ = hand.landmarks_norm[a_idx]
		bx, by, _ = hand.landmarks_norm[b_idx]
		return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

	index = hand.landmarks_norm[8]
	index_pip = hand.landmarks_norm[6]
	middle = hand.landmarks_norm[12]
	middle_pip = hand.landmarks_norm[10]
	ring = hand.landmarks_norm[16]
	ring_pip = hand.landmarks_norm[14]
	pinky = hand.landmarks_norm[20]
	pinky_pip = hand.landmarks_norm[18]

	index_up = index[1] < index_pip[1]
	middle_up = middle[1] < middle_pip[1]
	ring_up = ring[1] < ring_pip[1]
	pinky_up = pinky[1] < pinky_pip[1]
	thumb_open = _dist(4, 0) > (_dist(3, 0) * 1.08)

	return index_up and not middle_up and not ring_up and not pinky_up and not thumb_open


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
	x1, y1, x2, y2 = rect
	return x1 <= x <= x2 and y1 <= y <= y2


def _service_running() -> bool:
	if not CONTROL_PID_FILE.exists():
		return False
	try:
		pid = int(CONTROL_PID_FILE.read_text(encoding="utf-8").strip())
	except Exception:
		return False
	return pid > 0


def _start_service() -> None:
	subprocess.run([str(START_CONTROL_SCRIPT), "start"], check=False)


def _stop_service() -> None:
	subprocess.run([str(START_CONTROL_SCRIPT), "stop"], check=False)


def _draw_buttons(panel, real_mode: bool, controls_enabled: bool, service_running: bool, input_backend: str) -> None:
	real_color = (0, 180, 0) if real_mode else (60, 60, 200)
	ctrl_color = (0, 140, 0) if controls_enabled else (0, 90, 180)
	stop_color = (30, 30, 30)

	cv2.rectangle(panel, (REAL_BTN[0], REAL_BTN[1]), (REAL_BTN[2], REAL_BTN[3]), real_color, -1)
	cv2.rectangle(panel, (CTRL_BTN[0], CTRL_BTN[1]), (CTRL_BTN[2], CTRL_BTN[3]), ctrl_color, -1)
	cv2.rectangle(panel, (STOP_BTN[0], STOP_BTN[1]), (STOP_BTN[2], STOP_BTN[3]), stop_color, -1)

	mode_text = "REAL MODE: ON" if real_mode else "REAL MODE: OFF"
	cv2.putText(panel, mode_text, (REAL_BTN[0] + 10, REAL_BTN[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
	ctrl_text = "CONTROLS: ON" if controls_enabled else "CONTROLS: OFF"
	cv2.putText(panel, ctrl_text, (CTRL_BTN[0] + 10, CTRL_BTN[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(panel, "STOP APP", (STOP_BTN[0] + 80, STOP_BTN[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.rectangle(panel, (SETTINGS_BTN[0], SETTINGS_BTN[1]), (SETTINGS_BTN[2], SETTINGS_BTN[3]), (80, 80, 80), -1)
	cv2.putText(panel, "SETTINGS", (SETTINGS_BTN[0] + 90, SETTINGS_BTN[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

	status = "CONTROLS ACTIVE" if controls_enabled else "CONTROLS OFF"
	color = (0, 220, 0) if controls_enabled else (0, 160, 230)
	cv2.putText(panel, status, (12, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
	cv2.putText(panel, "M: mode  C: controls  Q: quit", (12, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
	service_text = "SERVICE: RUNNING" if service_running else "SERVICE: STOPPED"
	service_color = (0, 220, 0) if service_running else (0, 0, 220)
	cv2.putText(panel, service_text, (12, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.48, service_color, 1, cv2.LINE_AA)
	backend_color = (0, 220, 0) if input_backend != "disabled" else (0, 0, 220)
	cv2.putText(panel, f"INPUT: {input_backend.upper()}", (12, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.42, backend_color, 1, cv2.LINE_AA)


def _draw_gesture_log(panel, current_gesture: str | None, history: list[str]) -> None:
	if current_gesture:
		cv2.putText(panel, f"LAST: {current_gesture}", (12, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
	else:
		cv2.putText(panel, "LAST: -", (12, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

	cv2.putText(panel, "GESTURE LOG", (12, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
	y = 324
	for entry in history[:3]:
		cv2.putText(panel, entry, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
		y += 20


def run(show_window: bool = True, enable_hotkeys: bool = True) -> None:
	config = AppConfig()
	service_mode = not show_window

	camera = CameraStream(
		camera_index=config.camera_index,
		frame_width=config.frame_width,
		frame_height=config.frame_height,
	)
	camera_active = False

	tracker = HandTracker(
		max_num_hands=config.max_num_hands,
		model_complexity=config.model_complexity,
		min_detection_confidence=config.min_detection_confidence,
		min_tracking_confidence=config.min_tracking_confidence,
	)
	gesture_engine = GestureEngine(
		click_pinch_threshold=config.click_pinch_threshold,
		click_cooldown_ms=config.click_cooldown_ms,
		scroll_step=config.scroll_step,
		scroll_trigger_delta=config.scroll_trigger_delta,
		scroll_cooldown_ms=config.scroll_cooldown_ms,
		tab_trigger_delta=config.tab_trigger_delta,
		tab_cooldown_ms=config.tab_cooldown_ms,
		tab_gesture_cooldown_ms=config.tab_gesture_cooldown_ms,
		zoom_trigger_delta=config.zoom_trigger_delta,
		zoom_cooldown_ms=config.zoom_cooldown_ms,
		horizontal_align_tolerance=config.horizontal_align_tolerance,
		vertical_align_tolerance=config.vertical_align_tolerance,
		finger_span_min=config.finger_span_min,
	)
	cursor_controller = CursorController(
		smoothing_alpha=config.smoothing_alpha,
		move_interval_ms=config.cursor_move_interval_ms,
		min_move_pixels=config.cursor_min_move_pixels,
	)
	dispatcher = ActionDispatcher(cursor_controller)
	overlay = Overlay()
	hotkeys = HotkeyController(enabled=enable_hotkeys)
	hotkeys.start()
	gesture_counts: dict[str, int] = {}
	gesture_history: deque[str] = deque(maxlen=config.gesture_log_limit)
	last_event_ms: dict[str, float] = {}
	last_gesture_display: str | None = None
	last_gesture_until_ms = 0.0
	virtual_keyboard = None
	keyboard_open_in_browser = False
	keyboard_file_url: str | None = None
	last_keyboard_click_ms = 0.0
	keyboard_click_cooldown = config.click_cooldown_ms
	no_frame_count = 0
	last_camera_warning: str | None = None
	is_wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

	ui_state = {
		"real_mode": service_mode,
		"controls_enabled": True,
		"exit": False,
	}
	if service_mode:
		camera_active = camera.start()
		if not camera_active:
			raise RuntimeError("Unable to open camera")
	elif not ui_state["real_mode"]:
		camera_active = camera.start()
		if not camera_active:
			last_camera_warning = "Unable to open camera"
	else:
		_start_service()

	camera_window_visible = False
	panel_window_visible = False

	if show_window:
		cv2.namedWindow(PANEL_NAME)
		cv2.namedWindow(config.window_name)
		camera_window_visible = True
		panel_window_visible = True

		# settings_request: set to True by panel mouse callback to open modal settings
		settings_request = False

		def _run_settings_modal() -> None:
			"""Run a modal settings editor. This stops the camera while open and
			applies settings only when user saves (press 's'). Press 'c' or Esc to cancel.
			"""
			nonlocal camera_active
			# save originals
			orig = {
				"click_pinch_threshold": gesture_engine.click_pinch_threshold,
				"scroll_trigger_delta": gesture_engine.scroll_trigger_delta,
				"scroll_cooldown_ms": gesture_engine.scroll_cooldown_ms,
				"tab_trigger_delta": gesture_engine.tab_trigger_delta,
				"tab_cooldown_ms": gesture_engine.tab_cooldown_ms,
				"tab_gesture_cooldown_ms": gesture_engine.tab_gesture_cooldown_ms,
				"gesture_repeat_delay_ms": config.gesture_repeat_delay_ms,
				"zoom_trigger_delta": gesture_engine.zoom_trigger_delta,
				"zoom_cooldown_ms": gesture_engine.zoom_cooldown_ms,
			}
			was_camera_active = camera_active
			if was_camera_active:
				camera.stop()
				camera_active = False

			cv2.namedWindow("Settings")
			# create trackbars (thresholds in thousandths)
			cv2.createTrackbar("click_thresh", "Settings", max(1, int(orig["click_pinch_threshold"] * 1000)), 200, lambda v: None)
			cv2.createTrackbar("scroll_trigger", "Settings", max(1, int(orig["scroll_trigger_delta"] * 1000)), 500, lambda v: None)
			cv2.createTrackbar("scroll_cooldown", "Settings", int(orig["scroll_cooldown_ms"]), 3000, lambda v: None)
			cv2.createTrackbar("tab_trigger", "Settings", max(1, int(orig["tab_trigger_delta"] * 1000)), 500, lambda v: None)
			cv2.createTrackbar("tab_cooldown", "Settings", int(orig["tab_cooldown_ms"]), 3000, lambda v: None)
			cv2.createTrackbar("tab_gesture_cooldown", "Settings", int(orig["tab_gesture_cooldown_ms"]), 5000, lambda v: None)
			cv2.createTrackbar("gesture_repeat_delay", "Settings", int(orig["gesture_repeat_delay_ms"]), 5000, lambda v: None)
			cv2.createTrackbar("zoom_trigger", "Settings", max(1, int(orig["zoom_trigger_delta"] * 1000)), 500, lambda v: None)
			cv2.createTrackbar("zoom_cooldown", "Settings", int(orig["zoom_cooldown_ms"]), 3000, lambda v: None)

			while True:
				# draw simple status image
				img = np.zeros((320, 640, 3), dtype=np.uint8)
				cv2.putText(img, "Settings — adjust trackbars, S=Save, C=Cancel", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
				# read trackbars and show current numeric values
				try:
					ct = cv2.getTrackbarPos("click_thresh", "Settings")
					scroll_tr = cv2.getTrackbarPos("scroll_trigger", "Settings")
					scroll_cd = cv2.getTrackbarPos("scroll_cooldown", "Settings")
					tab_tr = cv2.getTrackbarPos("tab_trigger", "Settings")
					tab_cd = cv2.getTrackbarPos("tab_cooldown", "Settings")
					tab_gcd = cv2.getTrackbarPos("tab_gesture_cooldown", "Settings")
					gr_d = cv2.getTrackbarPos("gesture_repeat_delay", "Settings")
					zoom_tr = cv2.getTrackbarPos("zoom_trigger", "Settings")
					zoom_cd = cv2.getTrackbarPos("zoom_cooldown", "Settings")
					txt_y = 60
					cv2.putText(img, f"click_thresh: {ct/1000.0:.3f}", (12, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
					cv2.putText(img, f"scroll_trigger: {scroll_tr/1000.0:.3f}  cooldown: {scroll_cd}ms", (12, txt_y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
					cv2.putText(img, f"tab_trigger: {tab_tr/1000.0:.3f}  tab_cd: {tab_cd}ms  tab_g_cd: {tab_gcd}ms", (12, txt_y+56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
					cv2.putText(img, f"gesture_repeat_delay: {gr_d}ms", (12, txt_y+84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
					cv2.putText(img, f"zoom_trigger: {zoom_tr/1000.0:.3f}  zoom_cd: {zoom_cd}ms", (12, txt_y+112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
				except Exception:
					pass
				cv2.imshow("Settings", img)
				key = cv2.waitKey(100) & 0xFF
				if key in (ord("s"), ord("S")):
					# apply values
					try:
						ct = cv2.getTrackbarPos("click_thresh", "Settings")
						scroll_tr = cv2.getTrackbarPos("scroll_trigger", "Settings")
						scroll_cd = cv2.getTrackbarPos("scroll_cooldown", "Settings")
						tab_tr = cv2.getTrackbarPos("tab_trigger", "Settings")
						tab_cd = cv2.getTrackbarPos("tab_cooldown", "Settings")
						tab_gcd = cv2.getTrackbarPos("tab_gesture_cooldown", "Settings")
						gr_d = cv2.getTrackbarPos("gesture_repeat_delay", "Settings")
						zoom_tr = cv2.getTrackbarPos("zoom_trigger", "Settings")
						zoom_cd = cv2.getTrackbarPos("zoom_cooldown", "Settings")
						gesture_engine.click_pinch_threshold = max(0.001, ct / 1000.0)
						gesture_engine.scroll_trigger_delta = max(0.001, scroll_tr / 1000.0)
						gesture_engine.scroll_cooldown_ms = int(scroll_cd)
						gesture_engine.tab_trigger_delta = max(0.001, tab_tr / 1000.0)
						gesture_engine.tab_cooldown_ms = int(tab_cd)
						gesture_engine.tab_gesture_cooldown_ms = int(tab_gcd)
						config.gesture_repeat_delay_ms = int(gr_d)
						gesture_engine.zoom_trigger_delta = max(0.001, zoom_tr / 1000.0)
						gesture_engine.zoom_cooldown_ms = int(zoom_cd)
					except Exception:
						pass
					break
				elif key in (ord("c"), ord("C"), 27):
					# cancel: restore originals
					gesture_engine.click_pinch_threshold = orig["click_pinch_threshold"]
					gesture_engine.scroll_trigger_delta = orig["scroll_trigger_delta"]
					gesture_engine.scroll_cooldown_ms = orig["scroll_cooldown_ms"]
					gesture_engine.tab_trigger_delta = orig["tab_trigger_delta"]
					gesture_engine.tab_cooldown_ms = orig["tab_cooldown_ms"]
					gesture_engine.tab_gesture_cooldown_ms = orig["tab_gesture_cooldown_ms"]
					config.gesture_repeat_delay_ms = orig["gesture_repeat_delay_ms"]
					gesture_engine.zoom_trigger_delta = orig["zoom_trigger_delta"]
					gesture_engine.zoom_cooldown_ms = orig["zoom_cooldown_ms"]
					break
			# close settings window
			try:
				cv2.destroyWindow("Settings")
			except Exception:
				pass
			# restart camera if it was active
			if was_camera_active:
				camera_active = camera.start()
				if not camera_active:
					last_camera_warning = "Unable to open camera"

		def _set_real_mode(enabled: bool) -> None:
			nonlocal camera_active
			if service_mode:
				ui_state["real_mode"] = True
				return
			if ui_state["real_mode"] == enabled:
				return
			ui_state["real_mode"] = enabled
			if enabled:
				if camera_active:
					camera.stop()
					camera_active = False
				_start_service()
			else:
				_stop_service()
				if not camera_active:
					camera_active = camera.start()
					if not camera_active:
						raise RuntimeError("Unable to open camera")

		def _on_mouse(event, x, y, _flags, _userdata):
			if event != cv2.EVENT_LBUTTONDOWN:
				return
			if _point_in_rect(x, y, REAL_BTN):
				_set_real_mode(not ui_state["real_mode"])
			elif _point_in_rect(x, y, CTRL_BTN):
				ui_state["controls_enabled"] = not ui_state["controls_enabled"]
			elif _point_in_rect(x, y, STOP_BTN):
				ui_state["exit"] = True
			elif _point_in_rect(x, y, SETTINGS_BTN):
				nonlocal settings_request
				settings_request = True

		cv2.setMouseCallback(PANEL_NAME, _on_mouse)

	prev_time = time.perf_counter()

	try:
		while True:
			# If the settings button was clicked, run modal settings (pauses camera)
			if show_window and settings_request:
				# reset request flag and run modal
				settings_request = False
				_run_settings_modal()

			now_ms = time.monotonic() * 1000.0
			if hotkeys.should_exit():
				break
			if hotkeys.consume_mode_toggle():
				if not service_mode:
					if show_window:
						_set_real_mode(not ui_state["real_mode"])
			if ui_state["exit"]:
				break

			service_running = _service_running()
			if show_window and ui_state["real_mode"] and not service_mode and not service_running:
				ui_state["real_mode"] = False
				if not camera_active:
					camera_active = camera.start()
					if not camera_active:
						raise RuntimeError("Unable to open camera")
			controls_enabled = ui_state["controls_enabled"]
			hands: list = []
			events = []
			cursor_point = None
			frame = np.zeros((config.frame_height, config.frame_width, 3), dtype=np.uint8)

			# if virtual keyboard is open, update it with current hands
			if virtual_keyboard is not None:
				try:
					virtual_keyboard.update(hands)
				except Exception:
					virtual_keyboard = None

			if service_mode or not ui_state["real_mode"] or not service_running:
				if not camera_active:
					camera_active = camera.start()
					if not camera_active:
						last_camera_warning = "Unable to open camera"
						if not show_window:
							time.sleep(0.05)
						continue
				frame = camera.read()
				if frame is None:
					no_frame_count += 1
					last_camera_warning = "No camera frames"
					if (no_frame_count % 120) == 0:
						camera.stop()
						camera_active = camera.start()
					frame = np.zeros((config.frame_height, config.frame_width, 3), dtype=np.uint8)
					if not show_window:
						time.sleep(0.01)
				else:
					no_frame_count = 0
					last_camera_warning = None

				if config.mirror_frame:
					frame = cv2.flip(frame, 1)

				if last_camera_warning is None:
					hands = tracker.process(frame)
					raw_events = gesture_engine.update(hands, now_ms)
					events = []
					for event in raw_events:
						last_ms = last_event_ms.get(event.name, 0.0)
						if (now_ms - last_ms) < config.gesture_repeat_delay_ms:
							continue
						last_event_ms[event.name] = now_ms
						# handle virtual keyboard open event in UI instead of dispatching
						if event.name == GestureEngine.OPEN_KEYBOARD:
							kb_path = _PROJECT_DIR / "hand_gesture_controller" / "ui" / "web_keyboard" / "index.html"
							# try xdg-open first to open file in existing browser
							try:
								res = subprocess.run(["xdg-open", str(kb_path)], check=False)
								if res.returncode == 0:
									keyboard_open_in_browser = True
									keyboard_file_url = kb_path.as_uri()
									# give browser a moment to open
									time.sleep(0.25)
								else:
									# fallback: try chromium directly
									try:
										subprocess.Popen(["chromium", "--new-window", str(kb_path)])
										keyboard_open_in_browser = True
										keyboard_file_url = kb_path.as_uri()
									except Exception:
										# final fallback: internal OpenCV keyboard
										if virtual_keyboard is None:
											virtual_keyboard = VirtualKeyboard()
							except Exception:
								# if xdg-open itself failed, fall back to internal keyboard
								if virtual_keyboard is None:
									virtual_keyboard = VirtualKeyboard()
							# record gesture but don't dispatch
							events.append(event)
						else:
							events.append(event)
						gesture_counts[event.name] = gesture_counts.get(event.name, 0) + 1
						label = {
							"OPEN_TAB": "Open Tab",
							"CLOSE_TAB": "Close Tab",
							"NEXT_TAB": "Next Tab",
							"PREV_TAB": "Previous Tab",
							"LEFT_CLICK": "Left Click",
							"RIGHT_CLICK": "Right Click",
							"SCROLL_UP": "Scroll Up",
							"SCROLL_DOWN": "Scroll Down",
							"ZOOM_IN": "Zoom In",
							"ZOOM_OUT": "Zoom Out",
						}.get(event.name, event.name.replace("_", " ").title())
						log_line = f"{label} x{gesture_counts[event.name]}"
						gesture_history.appendleft(log_line)
						last_gesture_display = log_line
						last_gesture_until_ms = now_ms + config.gesture_display_ms
					if controls_enabled:
						dispatcher.dispatch(events)

					# Settings are applied by a modal; no live trackbar polling here.

					# If keyboard page opened in browser, translate index finger to system cursor and clicks
					if keyboard_open_in_browser and hands:
						primary_hand = next((h for h in hands if h.handedness == "Right"), hands[0])
						# move system cursor to index finger
						ix, iy, _ = primary_hand.landmarks_norm[8]
						if config.enable_system_cursor:
							cursor_controller.move_from_normalized(ix, iy)
						# detect index bend (tip near pip) and click
						ax, ay, _ = primary_hand.landmarks_norm[8]
						bx, by, _ = primary_hand.landmarks_norm[6]
						dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
						now_click_ms = now_ms
						if dist < 0.03 and (now_click_ms - last_keyboard_click_ms) >= keyboard_click_cooldown:
							cursor_controller.click_left()
							last_keyboard_click_ms = now_click_ms

					if hands:
						primary_hand = next((h for h in hands if h.handedness == "Right"), hands[0])
						if _pointer_active(primary_hand):
							ix, iy, _ = primary_hand.landmarks_norm[8]
							h, w = frame.shape[:2]
							cursor_point = (int(ix * w), int(iy * h))

							if config.enable_system_cursor and controls_enabled:
								cursor_controller.move_from_normalized(ix, iy)

			current_time = time.perf_counter()
			delta = max(current_time - prev_time, 1e-6)
			fps = 1.0 / delta
			prev_time = current_time

			if show_window:
				if ui_state["real_mode"] and not service_mode:
					if camera_window_visible:
						cv2.destroyWindow(config.window_name)
						camera_window_visible = False
				else:
					if not panel_window_visible:
						cv2.namedWindow(PANEL_NAME)
						cv2.setMouseCallback(PANEL_NAME, _on_mouse)
						panel_window_visible = True
					if not camera_window_visible:
						cv2.namedWindow(config.window_name)
						camera_window_visible = True
					current_gesture = last_gesture_display if now_ms <= last_gesture_until_ms else None
					display_frame = overlay.draw(
						frame,
						hands=hands,
						fps=fps,
						current_gesture=current_gesture,
						cursor_point=cursor_point,
					)
					if last_camera_warning is not None:
						cv2.putText(display_frame, f"CAMERA: {last_camera_warning}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.putText(display_frame, "Check camera permissions/device in use", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 180, 255), 1, cv2.LINE_AA)
					cv2.imshow(config.window_name, display_frame)

				if not panel_window_visible:
					cv2.namedWindow(PANEL_NAME)
					cv2.setMouseCallback(PANEL_NAME, _on_mouse)
					panel_window_visible = True

				panel = np.zeros((390, 420, 3), dtype=np.uint8)
				_draw_buttons(panel, ui_state["real_mode"], controls_enabled, service_running, cursor_controller.backend_name)
				_draw_gesture_log(panel, last_gesture_display if now_ms <= last_gesture_until_ms else None, list(gesture_history))
				# show system cursor coordinates when web keyboard is open (helpful for debugging mapping)
				if keyboard_open_in_browser and hands:
					try:
						screen_w, screen_h = cursor_controller._backend.screen_size() if cursor_controller._backend is not None else (0, 0)
						primary_hand = next((h for h in hands if h.handedness == "Right"), hands[0])
						ix, iy, _ = primary_hand.landmarks_norm[8]
						sx = int(ix * screen_w) if screen_w else 0
						sy = int(iy * screen_h) if screen_h else 0
						cv2.putText(panel, f"SYS CURSOR: {sx},{sy}", (12, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
					except Exception:
						pass
				if cursor_controller.backend_name == "disabled":
					cv2.putText(panel, "Tip: install xdotool or run X11", (12, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 220), 1, cv2.LINE_AA)
				elif is_wayland and cursor_controller.backend_name == "xdotool":
					cv2.putText(panel, "Wayland browser may ignore xdotool", (12, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 160, 255), 1, cv2.LINE_AA)
				cv2.imshow(PANEL_NAME, panel)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
				elif key in (ord("m"), ord("M")):
					if not service_mode:
						_set_real_mode(not ui_state["real_mode"])
				elif key in (ord("c"), ord("C")):
					ui_state["controls_enabled"] = not ui_state["controls_enabled"]
	finally:
		hotkeys.stop()
		_stop_service()
		camera.stop()
		tracker.close()
		if show_window:
			try:
				cv2.destroyWindow(PANEL_NAME)
			except Exception:
				pass
			cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--headless", action="store_true", help="Run without preview window")
	parser.add_argument("--no-hotkeys", action="store_true", help="Disable global hotkeys")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(show_window=not args.headless, enable_hotkeys=not args.no_hotkeys)
