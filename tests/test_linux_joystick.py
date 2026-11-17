import struct

import pytest

from supermimic.runners.linux_joystick import LinuxJoystick


def test_joystick_axes_deadzone_and_initial_events(monkeypatch):
  import supermimic.runners.linux_joystick as joystick

  events = iter([struct.pack("IhBB", 0, -32767, 0x82, 1),
                 struct.pack("IhBB", 0, 1000, 0x02, 0),
                 struct.pack("IhBB", 0, 32767, 0x02, 3)])
  monkeypatch.setattr(joystick.os, "open", lambda *args: 12)
  def read(*args):
    try:
      return next(events)
    except StopIteration:
      raise BlockingIOError from None
  monkeypatch.setattr(joystick.os, "read", read)
  stick = LinuxJoystick("/dev/input/js0")
  assert stick.read() == pytest.approx((0.6, 0.0, -0.5))
