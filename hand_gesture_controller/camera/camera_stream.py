from __future__ import annotations

"""Manages webcam capture and frame retrieval.
Handles camera initialization, frame acquisition, and camera readiness checks.
"""

import fcntl
import os
import select
import shutil
import subprocess
import time

import cv2
import numpy as np


class CameraStream:
	def __init__(self, camera_index: int, frame_width: int, frame_height: int) -> None:
		self.camera_index = camera_index
		self.frame_width = frame_width
		self.frame_height = frame_height
		self._capture: cv2.VideoCapture | None = None
		self._first_frame = None

		self._rpicam_process: subprocess.Popen | None = None
		self._mjpeg_buffer = bytearray()

	def _try_start_opencv(self) -> bool:
		self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
		if not self._capture.isOpened():
			self._capture.release()
			self._capture = cv2.VideoCapture(self.camera_index)
		if not self._capture.isOpened():
			return False

		self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
		self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

		# Some Pi camera nodes open but do not return frames immediately.
		for _ in range(15):
			ok, frame = self._capture.read()
			if ok and frame is not None:
				self._first_frame = frame
				return True
			time.sleep(0.01)

		self._capture.release()
		self._capture = None
		return False

	def _start_rpicam(self) -> bool:
		if shutil.which("rpicam-vid") is None:
			return False

		cmd = [
			"rpicam-vid",
			"-t",
			"0",
			"--codec",
			"mjpeg",
			"--inline",
			"-n",
			"--width",
			str(self.frame_width),
			"--height",
			str(self.frame_height),
			"-o",
			"-",
		]

		self._rpicam_process = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL,
			bufsize=0,
		)

		if self._rpicam_process.stdout is not None:
			fd = self._rpicam_process.stdout.fileno()
			flags = fcntl.fcntl(fd, fcntl.F_GETFL)
			fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

		first = self._read_rpicam_frame(timeout_s=2.0)
		if first is None:
			self.stop()
			return False

		self._first_frame = first
		return True

	def _extract_latest_mjpeg_frame(self):
		latest_frame = None

		while True:
			start = self._mjpeg_buffer.find(b"\xff\xd8")
			if start < 0:
				if len(self._mjpeg_buffer) > 1024:
					del self._mjpeg_buffer[:-2]
				break

			if start > 0:
				del self._mjpeg_buffer[:start]

			end = self._mjpeg_buffer.find(b"\xff\xd9", 2)
			if end < 0:
				break

			jpeg_data = bytes(self._mjpeg_buffer[: end + 2])
			del self._mjpeg_buffer[: end + 2]

			frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
			if frame is not None:
				latest_frame = frame

		return latest_frame

	def _read_rpicam_frame(self, timeout_s: float = 0.05):
		if self._rpicam_process is None or self._rpicam_process.stdout is None:
			return None

		stream = self._rpicam_process.stdout
		latest_frame = self._extract_latest_mjpeg_frame()

		deadline = time.monotonic() + timeout_s
		while time.monotonic() < deadline:
			if self._rpicam_process.poll() is not None:
				break

			wait_s = max(0.0, deadline - time.monotonic())
			ready, _, _ = select.select([stream], [], [], wait_s)
			if not ready:
				break

			while True:
				try:
					chunk = os.read(stream.fileno(), 65536)
				except BlockingIOError:
					break

				if not chunk:
					break

				self._mjpeg_buffer.extend(chunk)
				if len(self._mjpeg_buffer) > 4 * 1024 * 1024:
					del self._mjpeg_buffer[: len(self._mjpeg_buffer) - 1024 * 1024]

			frame = self._extract_latest_mjpeg_frame()
			if frame is not None:
				latest_frame = frame
				if not select.select([stream], [], [], 0)[0]:
					break

		return latest_frame

	def start(self) -> bool:
		if self._try_start_opencv():
			return True
		return self._start_rpicam()

	def read(self):
		if self._first_frame is not None:
			frame = self._first_frame
			self._first_frame = None
			return frame

		if self._capture is None:
			return self._read_rpicam_frame(timeout_s=0.03)

		ok, frame = self._capture.read()
		if ok and frame is not None:
			return frame
		return None

	def stop(self) -> None:
		if self._capture is not None:
			self._capture.release()
			self._capture = None

		if self._rpicam_process is not None:
			try:
				self._rpicam_process.terminate()
				self._rpicam_process.wait(timeout=0.5)
			except Exception:
				try:
					self._rpicam_process.kill()
				except Exception:
					pass
			self._rpicam_process = None

		self._mjpeg_buffer.clear()
		self._first_frame = None
