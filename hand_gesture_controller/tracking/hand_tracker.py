from __future__ import annotations

"""Tracks hand landmarks using MediaPipe and converts them into TrackedHand objects.
Used by the gesture engine to detect hand poses and events.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp # type: ignore


@dataclass
class TrackedHand:
	handedness: str
	landmarks_norm: list[tuple[float, float, float]]
	landmarks_px: list[tuple[int, int]]


class HandTracker:
	def __init__(
		self,
		max_num_hands: int,
		model_complexity: int,
		min_detection_confidence: float,
		min_tracking_confidence: float,
	) -> None:
		self._mp_hands = mp.solutions.hands
		self._hands = self._mp_hands.Hands(
			model_complexity=model_complexity,
			max_num_hands=max_num_hands,
			min_detection_confidence=min_detection_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)

	def process(self, frame_bgr) -> list[TrackedHand]:
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		frame_rgb.flags.writeable = False
		result = self._hands.process(frame_rgb)
		frame_rgb.flags.writeable = True
		if not result.multi_hand_landmarks:
			return []

		height, width = frame_bgr.shape[:2]
		output: list[TrackedHand] = []

		for i, landmarks in enumerate(result.multi_hand_landmarks):
			if result.multi_handedness and i < len(result.multi_handedness):
				handedness = result.multi_handedness[i].classification[0].label
			else:
				handedness = "Unknown"

			norm_points: list[tuple[float, float, float]] = []
			px_points: list[tuple[int, int]] = []

			for landmark in landmarks.landmark:
				norm_points.append((landmark.x, landmark.y, landmark.z))
				x = int(landmark.x * width)
				y = int(landmark.y * height)
				px_points.append((x, y))

			output.append(
				TrackedHand(
					handedness=handedness,
					landmarks_norm=norm_points,
					landmarks_px=px_points,
				)
			)

		return output

	def close(self) -> None:
		self._hands.close()
