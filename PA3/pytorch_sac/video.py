import imageio
import os
import numpy as np
import sys

import utils

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            # Support both old Gym (env.render(mode=..., height=...)) and
            # new Gymnasium (env created with render_mode='rgb_array' and
            # env.render() called without kwargs). Be defensive: only
            # append frames that are valid image arrays (ndim >= 2).
            frame = None
            try:
                frame = env.render(mode='rgb_array',
                                   height=self.height,
                                   width=self.width,
                                   camera_id=self.camera_id)
            except TypeError:
                try:
                    frame = env.render()
                except TypeError:
                    try:
                        frame = env.render(mode='rgb_array')
                    except Exception:
                        frame = None

            # Normalize frame: find first ndarray with >=2 dims if wrapped
            valid_frame = None
            if frame is None:
                valid_frame = None
            elif hasattr(frame, 'ndim') and getattr(frame, 'ndim', 0) >= 2:
                valid_frame = frame
            elif isinstance(frame, (list, tuple)) and len(frame) > 0:
                for item in frame:
                    if hasattr(item, 'ndim') and getattr(item, 'ndim', 0) >= 2:
                        valid_frame = item
                        break

            if valid_frame is not None:
                # convert to uint8 if needed
                try:
                    arr = np.asarray(valid_frame)
                    if arr.dtype != np.uint8:
                        # scale floats in [0,1] to [0,255]
                        if np.issubdtype(arr.dtype, np.floating):
                            arr = (255 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                    self.frames.append(arr)
                except Exception:
                    pass

    def save(self, file_name):
        if self.enabled:
            if len(self.frames) == 0:
                return
            path = os.path.join(self.save_dir, file_name)
            try:
                frames = np.asarray(self.frames)
                # ensure uint8
                if frames.dtype != np.uint8:
                    if np.issubdtype(frames.dtype, np.floating):
                        frames = (255 * np.clip(frames, 0.0, 1.0)).astype(np.uint8)
                    else:
                        frames = frames.astype(np.uint8)
                imageio.mimsave(path, frames, fps=self.fps)
            except Exception:
                # if saving fails, skip silently to avoid crashing training
                return
