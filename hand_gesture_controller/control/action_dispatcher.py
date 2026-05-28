from __future__ import annotations

"""Maps gesture events to cursor controller methods.
Routes gesture actions to the appropriate input commands.
"""

from hand_gesture_controller.control.cursor_controller import CursorController
from hand_gesture_controller.gesture.gesture_engine import GestureEngine, GestureEvent


class ActionDispatcher:
	def __init__(self, cursor_controller: CursorController) -> None:
		self.cursor_controller = cursor_controller

	def dispatch(self, events: list[GestureEvent]) -> None:
		for event in events:
			if event.name == GestureEngine.LEFT_CLICK:
				self.cursor_controller.click_left()
			elif event.name == GestureEngine.RIGHT_CLICK:
				self.cursor_controller.click_right()
			elif event.name == GestureEngine.SCROLL_UP:
				self.cursor_controller.scroll(event.value or 100)
			elif event.name == GestureEngine.SCROLL_DOWN:
				self.cursor_controller.scroll(-(event.value or 100))
			elif event.name == GestureEngine.NEXT_TAB:
				self.cursor_controller.next_tab()
			elif event.name == GestureEngine.PREV_TAB:
				self.cursor_controller.prev_tab()
			elif event.name == GestureEngine.OPEN_TAB:
				self.cursor_controller.open_tab()
			elif event.name == GestureEngine.CLOSE_TAB:
				self.cursor_controller.close_tab()
			elif event.name == GestureEngine.ZOOM_IN:
				self.cursor_controller.zoom_in()
			elif event.name == GestureEngine.ZOOM_OUT:
				self.cursor_controller.zoom_out()
