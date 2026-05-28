from __future__ import annotations

"""Interprets tracked hand landmarks as gesture events.
Generates click, scroll, tab, and zoom events for the action dispatcher.
"""

from dataclasses import dataclass
from math import atan2, degrees, sqrt

from hand_gesture_controller.tracking.hand_tracker import TrackedHand


@dataclass
class GestureEvent:
	name: str
	confidence: float = 1.0
	value: int = 0


class GestureEngine:
	LEFT_CLICK = "LEFT_CLICK"
	RIGHT_CLICK = "RIGHT_CLICK"
	SCROLL_UP = "SCROLL_UP"
	SCROLL_DOWN = "SCROLL_DOWN"
	NEXT_TAB = "NEXT_TAB"
	PREV_TAB = "PREV_TAB"
	OPEN_TAB = "OPEN_TAB"
	CLOSE_TAB = "CLOSE_TAB"
	OPEN_KEYBOARD = "OPEN_KEYBOARD"
	ZOOM_IN = "ZOOM_IN"
	ZOOM_OUT = "ZOOM_OUT"

	def __init__(
		self,
		click_pinch_threshold: float,
		click_cooldown_ms: int,
		scroll_step: int,
		scroll_trigger_delta: float,
		scroll_cooldown_ms: int,
		tab_trigger_delta: float,
		tab_cooldown_ms: int,
		tab_gesture_cooldown_ms: int,
		zoom_trigger_delta: float,
		zoom_cooldown_ms: int,
		horizontal_align_tolerance: float,
		vertical_align_tolerance: float,
		finger_span_min: float,
	) -> None:
		self.click_pinch_threshold = click_pinch_threshold
		self.click_cooldown_ms = click_cooldown_ms
		self.scroll_step = scroll_step
		self.scroll_trigger_delta = scroll_trigger_delta
		self.scroll_cooldown_ms = scroll_cooldown_ms
		self.tab_trigger_delta = tab_trigger_delta
		self.tab_cooldown_ms = tab_cooldown_ms
		self.tab_gesture_cooldown_ms = tab_gesture_cooldown_ms
		self.zoom_trigger_delta = zoom_trigger_delta
		self.zoom_cooldown_ms = zoom_cooldown_ms
		self.horizontal_align_tolerance = horizontal_align_tolerance
		self.vertical_align_tolerance = vertical_align_tolerance
		self.finger_span_min = finger_span_min

		self._pinch_state = {"Left": False, "Right": False}
		self._last_click_ms = {"Left": 0.0, "Right": 0.0}

		self._scroll_anchor: dict[str, float] = {}
		self._last_scroll_ms: dict[str, float] = {}

		self._tab_anchor: dict[str, float] = {}
		self._last_tab_ms: dict[str, float] = {}
		self._last_tab_gesture_ms = 0.0

		self._zoom_distance: float | None = None
		self._last_zoom_ms = 0.0

	@staticmethod
	def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
		return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	def _fingers_extended(self, hand: TrackedHand) -> tuple[bool, tuple[float, float]]:
		tip_ids = [8, 12, 16, 20]
		pip_ids = [6, 10, 14, 18]

		extended = []
		for tip_id, pip_id in zip(tip_ids, pip_ids):
			tip_dist = self._distance_idx(hand, tip_id, 0)
			pip_dist = self._distance_idx(hand, pip_id, 0)
			extended.append(tip_dist > pip_dist * 1.05)

		if not all(extended):
			return False, (0.0, 0.0)

		tips = [hand.landmarks_norm[i] for i in tip_ids]
		xs = [p[0] for p in tips]
		ys = [p[1] for p in tips]
		center = (sum(xs) / 4.0, sum(ys) / 4.0)
		return True, center

	@staticmethod
	def _distance_idx(hand: TrackedHand, idx_a: int, idx_b: int) -> float:
		ax, ay, _ = hand.landmarks_norm[idx_a]
		bx, by, _ = hand.landmarks_norm[idx_b]
		return sqrt((ax - bx) ** 2 + (ay - by) ** 2)

	def _fingers_pose(self, hand: TrackedHand) -> tuple[bool, bool, tuple[float, float]]:
		extended, center = self._fingers_extended(hand)
		if not extended:
			return False, False, (0.0, 0.0)

		tip_ids = [8, 12, 16, 20]
		tips = [hand.landmarks_norm[i] for i in tip_ids]
		xs = [p[0] for p in tips]
		ys = [p[1] for p in tips]
		x_span = max(xs) - min(xs)
		y_span = max(ys) - min(ys)

		horizontal = y_span <= self.horizontal_align_tolerance and x_span >= self.finger_span_min
		vertical = x_span <= self.vertical_align_tolerance and y_span >= self.finger_span_min
		return horizontal, vertical, center

	def _hand_line_orientation(self, hand: TrackedHand) -> tuple[str | None, tuple[float, float]]:
		extended, center = self._fingers_extended(hand)
		if not extended:
			return None, (0.0, 0.0)

		tip_ids = [8, 12, 16, 20]
		tips = [hand.landmarks_norm[i] for i in tip_ids]
		xs = [p[0] for p in tips]
		ys = [p[1] for p in tips]
		mean_x = sum(xs) / 4.0
		mean_y = sum(ys) / 4.0
		cov_xx = sum((x - mean_x) ** 2 for x in xs)
		cov_yy = sum((y - mean_y) ** 2 for y in ys)
		cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
		if cov_xx + cov_yy < 1e-6:
			return None, center

		angle = 0.5 * atan2(2.0 * cov_xy, cov_xx - cov_yy)
		angle_deg = degrees(angle)
		angle_tol = 30.0

		if abs(angle_deg) <= angle_tol or abs(abs(angle_deg) - 180.0) <= angle_tol:
			return "horizontal", center
		if abs(abs(angle_deg) - 90.0) <= angle_tol:
			return "vertical", center
		if abs(angle_deg - 45.0) <= angle_tol:
			return "diag_pos", center
		if abs(angle_deg + 45.0) <= angle_tol:
			return "diag_neg", center
		return None, center

	def _both_hands_shape(self, left_hand: TrackedHand | None, right_hand: TrackedHand | None) -> str | None:
		if left_hand is None or right_hand is None:
			return None

		left_orient, left_center = self._hand_line_orientation(left_hand)
		right_orient, right_center = self._hand_line_orientation(right_hand)
		if left_orient is None or right_orient is None:
			return None

		distance = self._distance(left_center, right_center)
		if distance > 0.24:
			return None

		if {left_orient, right_orient} == {"horizontal", "vertical"}:
			return self.OPEN_TAB

		if {left_orient, right_orient} == {"diag_pos", "diag_neg"}:
			return self.CLOSE_TAB

		return None

	def _index_only_pose(self, hand: TrackedHand) -> tuple[bool, tuple[float, float]]:
		"""Return (True, center) when only the index finger is extended for the hand.
		Used to detect the two-index-fingers gesture to open the virtual keyboard.
		"""
		# tip ids: 8=index,12=middle,16=ring,20=pinky; pip ids: 6,10,14,18
		tip_ids = [8, 12, 16, 20]
		pip_ids = [6, 10, 14, 18]
		extended = []
		for tip_id, pip_id in zip(tip_ids, pip_ids):
			tip_dist = self._distance_idx(hand, tip_id, 0)
			pip_dist = self._distance_idx(hand, pip_id, 0)
			extended.append(tip_dist > pip_dist * 1.05)

		# Only index should be extended
		if extended[0] and not any(extended[1:]):
			tip = hand.landmarks_norm[8]
			return True, (tip[0], tip[1])
		return False, (0.0, 0.0)

	@staticmethod
	def _distance_idx(hand: TrackedHand, idx_a: int, idx_b: int) -> float:
		ax, ay, _ = hand.landmarks_norm[idx_a]
		bx, by, _ = hand.landmarks_norm[idx_b]
		return sqrt((ax - bx) ** 2 + (ay - by) ** 2)

	def _update_click(
		self,
		hand: TrackedHand | None,
		hand_name: str,
		event_name: str,
		now_ms: float,
		events: list[GestureEvent],
		allow_click: bool,
	) -> None:
		if hand is None or not allow_click:
			self._pinch_state[hand_name] = False
			return

		thumb_x, thumb_y, _ = hand.landmarks_norm[4]
		index_x, index_y, _ = hand.landmarks_norm[8]
		distance = self._distance((thumb_x, thumb_y), (index_x, index_y))
		pinched_now = distance < self.click_pinch_threshold

		if pinched_now and not self._pinch_state[hand_name]:
			if (now_ms - self._last_click_ms[hand_name]) >= self.click_cooldown_ms:
				events.append(GestureEvent(name=event_name, confidence=1.0))
				self._last_click_ms[hand_name] = now_ms

		self._pinch_state[hand_name] = pinched_now

	def _update_scroll_and_tab(
		self,
		hand_key: str,
		horizontal_pose: bool,
		vertical_pose: bool,
		center: tuple[float, float],
		now_ms: float,
		events: list[GestureEvent],
	) -> None:
		if horizontal_pose:
			anchor = self._scroll_anchor.get(hand_key)
			if anchor is None:
				self._scroll_anchor[hand_key] = center[1]
			else:
				delta = center[1] - anchor
				last_ms = self._last_scroll_ms.get(hand_key, 0.0)
				if (now_ms - last_ms) >= self.scroll_cooldown_ms:
					if delta <= -self.scroll_trigger_delta:
						events.append(GestureEvent(name=self.SCROLL_UP, value=self.scroll_step))
						self._scroll_anchor[hand_key] = center[1]
						self._last_scroll_ms[hand_key] = now_ms
					elif delta >= self.scroll_trigger_delta:
						events.append(GestureEvent(name=self.SCROLL_DOWN, value=self.scroll_step))
						self._scroll_anchor[hand_key] = center[1]
						self._last_scroll_ms[hand_key] = now_ms
		else:
			self._scroll_anchor.pop(hand_key, None)

		if vertical_pose:
			anchor = self._tab_anchor.get(hand_key)
			if anchor is None:
				self._tab_anchor[hand_key] = center[0]
			else:
				delta = center[0] - anchor
				last_ms = self._last_tab_ms.get(hand_key, 0.0)
				if (now_ms - last_ms) >= self.tab_cooldown_ms and (now_ms - self._last_tab_gesture_ms) >= self.tab_gesture_cooldown_ms:
					if delta >= self.tab_trigger_delta:
						events.append(GestureEvent(name=self.NEXT_TAB))
						self._tab_anchor[hand_key] = center[0]
						self._last_tab_ms[hand_key] = now_ms
						self._last_tab_gesture_ms = now_ms
					elif delta <= -self.tab_trigger_delta:
						events.append(GestureEvent(name=self.PREV_TAB))
						self._tab_anchor[hand_key] = center[0]
						self._last_tab_ms[hand_key] = now_ms
						self._last_tab_gesture_ms = now_ms
		else:
			self._tab_anchor.pop(hand_key, None)

	def _update_zoom(
		self,
		left_hand: TrackedHand | None,
		right_hand: TrackedHand | None,
		now_ms: float,
		events: list[GestureEvent],
	) -> None:
		if left_hand is None or right_hand is None:
			self._zoom_distance = None
			return

		left_x, left_y, _ = left_hand.landmarks_norm[0]
		right_x, right_y, _ = right_hand.landmarks_norm[0]
		distance = self._distance((left_x, left_y), (right_x, right_y))

		if self._zoom_distance is None:
			self._zoom_distance = distance
			return

		delta = distance - self._zoom_distance
		if (now_ms - self._last_zoom_ms) >= self.zoom_cooldown_ms:
			if delta >= self.zoom_trigger_delta:
				events.append(GestureEvent(name=self.ZOOM_OUT))
				self._zoom_distance = distance
				self._last_zoom_ms = now_ms
			elif delta <= -self.zoom_trigger_delta:
				events.append(GestureEvent(name=self.ZOOM_IN))
				self._zoom_distance = distance
				self._last_zoom_ms = now_ms
			else:
				self._zoom_distance = self._zoom_distance * 0.8 + distance * 0.2

	def update(self, hands: list[TrackedHand], now_ms: float) -> list[GestureEvent]:
		if not hands:
			self._pinch_state["Left"] = False
			self._pinch_state["Right"] = False
			self._scroll_anchor.clear()
			self._tab_anchor.clear()
			self._zoom_distance = None
			return []

		events: list[GestureEvent] = []
		left_hand = next((h for h in hands if h.handedness == "Left"), None)
		right_hand = next((h for h in hands if h.handedness == "Right"), None)

		# Check for two-index-fingers pose to open virtual keyboard (both hands with only index extended)
		if left_hand is not None and right_hand is not None:
			left_index_only, left_center = self._index_only_pose(left_hand)
			right_index_only, right_center = self._index_only_pose(right_hand)
			if left_index_only and right_index_only:
				# debounce with global tab/gesture cooldown
				if (now_ms - self._last_tab_gesture_ms) >= self.tab_gesture_cooldown_ms:
					events.append(GestureEvent(name=self.OPEN_KEYBOARD))
					self._last_tab_gesture_ms = now_ms
					return events

		shape_event = self._both_hands_shape(left_hand, right_hand)
		if shape_event is not None:
			if (now_ms - self._last_tab_gesture_ms) >= self.tab_gesture_cooldown_ms:
				events.append(GestureEvent(name=shape_event))
				self._last_tab_gesture_ms = now_ms
			return events

		hand_modes: dict[str, tuple[bool, bool, tuple[float, float]]] = {}
		left_key: str | None = None
		right_key: str | None = None
		any_line_mode = False

		for i, hand in enumerate(hands):
			hand_key = f"{hand.handedness}_{i}"
			hand_modes[hand_key] = self._fingers_pose(hand)
			horizontal_pose, vertical_pose, _ = hand_modes[hand_key]
			if horizontal_pose or vertical_pose:
				any_line_mode = True

			if hand is left_hand and left_key is None:
				left_key = hand_key
			if hand is right_hand and right_key is None:
				right_key = hand_key

		self._update_click(left_hand, "Left", self.LEFT_CLICK, now_ms, events, allow_click=not any_line_mode)
		self._update_click(right_hand, "Right", self.RIGHT_CLICK, now_ms, events, allow_click=not any_line_mode)

		for hand_key, mode in hand_modes.items():
			horizontal_pose, vertical_pose, center = mode
			self._update_scroll_and_tab(hand_key, horizontal_pose, vertical_pose, center, now_ms, events)

		if any_line_mode:
			self._zoom_distance = None
		else:
			self._update_zoom(left_hand, right_hand, now_ms, events)
		return events
