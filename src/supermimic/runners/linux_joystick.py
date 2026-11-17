"""Nonblocking Linux joystick velocity input; no external input package needed."""

import os
import struct


class LinuxJoystick:
  def __init__(self, path, forward_axis=1, lateral_axis=0, yaw_axis=3, deadzone=0.12,
               scales=(0.6, 0.3, 0.5)):
    if not 0 <= deadzone < 1 or len({forward_axis, lateral_axis, yaw_axis}) != 3:
      raise ValueError("invalid joystick axes or deadzone")
    self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    self.axes = {}
    self.indices = (forward_axis, lateral_axis, yaw_axis)
    self.deadzone, self.scales = deadzone, scales

  def read(self):
    while True:
      try:
        event = os.read(self.fd, 8)
      except BlockingIOError:
        break
      if not event:
        raise OSError("joystick disconnected")
      if len(event) != 8:
        raise OSError("short joystick event")
      _, value, kind, number = struct.unpack("IhBB", event)
      if kind & 0x7f == 0x02:
        self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
    result = []
    for axis, scale in zip(self.indices, self.scales, strict=True):
      value = -self.axes.get(axis, 0.0)
      value = 0.0 if abs(value) <= self.deadzone else (
        (abs(value) - self.deadzone) / (1 - self.deadzone) * (1 if value > 0 else -1)
      )
      result.append(value * scale)
    return tuple(result)

  def close(self):
    os.close(self.fd)
