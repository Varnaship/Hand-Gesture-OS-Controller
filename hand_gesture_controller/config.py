"""Defines application configuration and gesture tuning parameters.
Includes camera settings, cursor smoothing, gesture thresholds, and cooldown values.
"""
from dataclasses import dataclass


@dataclass
class AppConfig:
	camera_index: int = 0
	frame_width: int = 640
	frame_height: int = 480
	window_name: str = "Hand Gesture Controller"
	mirror_frame: bool = True

	max_num_hands: int = 2
	model_complexity: int = 0
	min_detection_confidence: float = 0.6
	min_tracking_confidence: float = 0.6

	enable_system_cursor: bool = True
	smoothing_alpha: float = 0.25
	cursor_move_interval_ms: int = 10
	cursor_min_move_pixels: float = 1.5

	click_pinch_threshold: float = 0.04
	click_cooldown_ms: int = 320

	scroll_step: int = 25
	scroll_trigger_delta: float = 0.03
	scroll_cooldown_ms: int = 260

	tab_trigger_delta: float = 0.04
	tab_cooldown_ms: int = 420
	tab_gesture_cooldown_ms: int = 1000

	zoom_trigger_delta: float = 0.05
	zoom_cooldown_ms: int = 320

	horizontal_align_tolerance: float = 0.08
	vertical_align_tolerance: float = 0.08
	finger_span_min: float = 0.12

	gesture_repeat_delay_ms: int = 420
	gesture_display_ms: int = 2500
	gesture_log_limit: int = 5
