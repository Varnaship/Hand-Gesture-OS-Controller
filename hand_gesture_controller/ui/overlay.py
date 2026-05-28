from __future__ import annotations

"""Draws overlay markers and gesture feedback on camera frames.
Used by the main app for visual debugging and gesture display.
"""

import cv2

from hand_gesture_controller.tracking.hand_tracker import TrackedHand


class Overlay:
	def draw(
		self,
		frame,
		hands: list[TrackedHand],
		fps: float,
		current_gesture: str | None = None,
		cursor_point: tuple[int, int] | None = None,
	):
		for hand in hands:
			for x, y in hand.landmarks_px:
				cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

			if hand.landmarks_px:
				hx, hy = hand.landmarks_px[0]
				cv2.putText(
					frame,
					hand.handedness,
					(hx + 8, hy - 8),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 0),
					2,
					cv2.LINE_AA,
				)

		if cursor_point is not None:
			cv2.circle(frame, cursor_point, 8, (0, 0, 255), 2)

		cv2.putText(
			frame,
			f"FPS: {fps:.1f}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)

		if current_gesture:
			cv2.putText(
				frame,
				f"Gesture: {current_gesture}",
				(10, 65),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

		return frame
