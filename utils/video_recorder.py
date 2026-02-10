"""Video recorder module for AutoStripe.

Toggle recording on/off via key press. Records front view and overhead view
to separate MP4 files in the videos/ directory.

Encoding and disk I/O run in a background thread so the main loop is not
blocked. Frames are queued; if the encoder falls behind, the oldest frames
are dropped to prevent memory buildup.
"""

import os
import time
import threading
from collections import deque

import cv2


# Max queued frames per view before dropping (~ 1 second at 20 FPS)
_MAX_QUEUE = 20


class VideoRecorder:
    """Toggle-able dual-view video recorder with background encoding."""

    def __init__(self, video_dir=None, fps=20.0):
        if video_dir is None:
            video_dir = os.path.join(
                os.path.dirname(__file__), '..', 'videos')
        self.video_dir = video_dir
        self.fps = fps
        self.recording = False

        self._writer_front = None
        self._writer_overhead = None
        self._front_size = None
        self._overhead_size = None

        # Background thread
        self._queue = deque(maxlen=_MAX_QUEUE * 2)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_thread = False
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True)
        self._thread.start()

    @property
    def is_recording(self):
        return self.recording

    def toggle(self, front_size=(1248, 384), overhead_size=(1800, 1600)):
        """Toggle recording on/off. Returns new recording state."""
        if self.recording:
            self.stop()
        else:
            self.start(front_size, overhead_size)
        return self.recording

    def start(self, front_size=(1248, 384), overhead_size=(1800, 1600)):
        """Start recording both views."""
        if self.recording:
            return

        os.makedirs(self.video_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self._front_size = front_size
        self._overhead_size = overhead_size

        front_path = os.path.join(
            self.video_dir, f'front_{timestamp}.avi')
        overhead_path = os.path.join(
            self.video_dir, f'overhead_{timestamp}.avi')

        self._writer_front = cv2.VideoWriter(
            front_path, fourcc, self.fps, front_size)
        self._writer_overhead = cv2.VideoWriter(
            overhead_path, fourcc, self.fps, overhead_size)

        self.recording = True
        print(f"\n{'='*50}")
        print(f"  Recording: ON")
        print(f"  Front:    {front_path}")
        print(f"  Overhead: {overhead_path}")
        print(f"{'='*50}\n")

    def stop(self):
        """Stop recording, flush queue, release writers."""
        if not self.recording:
            return

        self.recording = False

        # Flush remaining frames
        self._event.set()
        self._flush_queue()

        if self._writer_front is not None:
            self._writer_front.release()
            self._writer_front = None
        if self._writer_overhead is not None:
            self._writer_overhead.release()
            self._writer_overhead = None

        print(f"\n{'='*50}")
        print(f"  Recording: OFF (saved)")
        print(f"{'='*50}\n")

    def write_front(self, frame):
        """Queue a front view frame (BGR numpy array). Non-blocking."""
        if not self.recording or frame is None:
            return
        with self._lock:
            self._queue.append(('front', frame.copy()))
        self._event.set()

    def write_overhead(self, frame):
        """Queue an overhead view frame (BGR numpy array). Non-blocking."""
        if not self.recording or frame is None:
            return
        import numpy as np
        f = np.ascontiguousarray(frame[:, :, :3].copy())
        with self._lock:
            self._queue.append(('overhead', f))
        self._event.set()

    def release(self):
        """Release all resources (call in cleanup)."""
        self.stop()
        self._stop_thread = True
        self._event.set()
        self._thread.join(timeout=3.0)

    # --- Background thread ---

    def _writer_loop(self):
        """Background loop: wait for frames, encode and write to disk."""
        while not self._stop_thread:
            self._event.wait(timeout=0.5)
            self._event.clear()
            self._flush_queue()

    def _flush_queue(self):
        """Drain all queued frames and write them."""
        while True:
            with self._lock:
                if not self._queue:
                    break
                tag, frame = self._queue.popleft()

            if tag == 'front' and self._writer_front is not None:
                h, w = frame.shape[:2]
                fw, fh = self._front_size
                if w != fw or h != fh:
                    frame = cv2.resize(frame, self._front_size)
                self._writer_front.write(frame)

            elif tag == 'overhead' and self._writer_overhead is not None:
                h, w = frame.shape[:2]
                ow, oh = self._overhead_size
                if w != ow or h != oh:
                    frame = cv2.resize(frame, self._overhead_size)
                self._writer_overhead.write(frame)
