"""Direct serial connection to the GRBL controller - used only for jogging
during calibration, never for running actual burn jobs (that stays in
LightBurn; see the architecture note in export.py about why).

A serial port can only be held open by one program at a time, so LightBurn
must be disconnected from the machine while this is in use, and vice versa.

Jog targets/readouts use *work* coordinates (what LightBurn's on-screen
position display shows, i.e. relative to the currently active G54 origin),
not raw machine coordinates - GRBL's `$J=G90` jog command already
interprets X/Y in the active work coordinate system, and status reports are
converted from MPos using the WCO (work coordinate offset) GRBL reports
periodically.

Limit switches: if a jog drives into a limit switch, GRBL enters an ALARM
state and refuses every further motion command - including jogging away
from the switch - until it's explicitly cleared. This mirrors LightBurn's
own lightning-bolt (unlock, `$X`) and house (home, `$H`) buttons: unlock()
clears the alarm in place (position becomes approximate until re-homed),
home() runs the full homing cycle and re-zeros against the switches
properly. Every method that can hit this raises GrblAlarmError specifically
so the UI can point the user at those two recovery actions instead of just
failing silently.

The app's UI polls get_work_position()/get_state() on a timer *while* jog
buttons can also be clicked - both happen on the same shared connection
from different Flask request threads. Without synchronization, a status
poll's '?' and a jog's '$J=...ok' response can interleave on the wire and
get read by the wrong caller (looks like a jog randomly "not responding").
An RLock around every method serializes all serial I/O; it's reentrant
because methods like jog_relative() call get_state() internally while
already holding it.
"""
import re
import threading
import time

import serial
from serial.tools import list_ports as _list_ports

STATUS_RE = re.compile(r"<(\w+)\|MPos:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)")
WCO_RE = re.compile(r"WCO:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)")
ALARM_RE = re.compile(r"ALARM:(\d+)", re.IGNORECASE)

HOMING_TIMEOUT_S = 60.0


class GrblError(RuntimeError):
    pass


class GrblAlarmError(GrblError):
    """A limit switch (or other fault) put GRBL in ALARM state. Call
    unlock() or home() before trying to move again."""
    pass


def list_ports() -> list[str]:
    return [p.device for p in _list_ports.comports()]


class Grbl:
    def __init__(self, port: str, baud: int = 921600, timeout: float = 2.0):
        self.port = port
        self.baud = baud
        self.is_broken = False
        self._lock = threading.RLock()
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            raise GrblError(
                f"Couldn't open {port} - is LightBurn (or something else) "
                f"already connected to it? ({e})"
            )
        time.sleep(2)  # GRBL resets on a fresh serial connection; let it boot
        self._ser.reset_input_buffer()
        self._wco = (0.0, 0.0)
        self._refresh_wco()

    def _write(self, data: bytes) -> None:
        # Cheap CH340 adapters can drop the connection outright (e.g. from
        # electrical noise right as a limit switch trips), which surfaces
        # as pyserial's own SerialException - not our GrblError - so every
        # caller up the stack would crash uncaught unless this converts it.
        # is_broken lets the app notice and drop the stale connection
        # instead of repeating the same failing call forever.
        try:
            self._ser.write(data)
        except serial.SerialException as e:
            self.is_broken = True
            raise GrblError(f"Lost connection to the machine ({e}) - reconnect.")

    def _readline(self) -> str:
        try:
            return self._ser.readline().decode(errors="replace")
        except serial.SerialException as e:
            self.is_broken = True
            raise GrblError(f"Lost connection to the machine ({e}) - reconnect.")

    def _send_line(self, line: str) -> None:
        self._write((line.strip() + "\n").encode())

    def _read_response(self, timeout: float = 5.0) -> str:
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            chunk = self._readline()
            if not chunk:
                continue
            buf += chunk
            low = chunk.lower()
            if "alarm" in low:
                m = ALARM_RE.search(chunk)
                code = m.group(1) if m else "?"
                raise GrblAlarmError(
                    f"ALARM:{code} - likely hit a limit switch. Click Unlock "
                    f"or Home before jogging again."
                )
            if "ok" in low or "error" in low:
                return buf
        raise GrblError(f"No response from GRBL within {timeout}s")

    def _query_status(self, timeout: float = 2.0) -> str:
        self._write(b"?")
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline()
            if line.startswith("<"):
                return line
        raise GrblError("No status response from GRBL")

    def _refresh_wco(self) -> None:
        # WCO isn't in every status report - poll a few times if needed.
        for _ in range(5):
            line = self._query_status()
            m = WCO_RE.search(line)
            if m:
                self._wco = (float(m.group(1)), float(m.group(2)))
                return

    def get_state(self) -> str:
        """Idle / Run / Jog / Alarm / Home / Hold / Door / ..."""
        with self._lock:
            line = self._query_status()
            m = STATUS_RE.search(line)
            return m.group(1) if m else "Unknown"

    def get_work_position(self) -> tuple[float, float]:
        with self._lock:
            line = self._query_status()
            m = STATUS_RE.search(line)
            if not m:
                raise GrblError(f"Couldn't parse position from status report: {line!r}")
            wco_m = WCO_RE.search(line)
            if wco_m:
                self._wco = (float(wco_m.group(1)), float(wco_m.group(2)))
            mx, my = float(m.group(2)), float(m.group(3))
            return mx - self._wco[0], my - self._wco[1]

    def is_idle(self) -> bool:
        return self.get_state() == "Idle"

    def _check_not_alarmed(self) -> None:
        state = self.get_state()
        if state == "Alarm":
            raise GrblAlarmError(
                "GRBL is in ALARM state (likely a limit switch) - click "
                "Unlock or Home before jogging."
            )

    def jog_to(self, x_mm: float, y_mm: float, feed_mm_min: float = 1500) -> None:
        with self._lock:
            self._check_not_alarmed()
            self.laser_off()
            self._send_line(f"$J=G90 G21 X{x_mm:.3f} Y{y_mm:.3f} F{feed_mm_min:.0f}")
            self._read_response()
            self._wait_until_idle()

    def jog_relative(self, dx_mm: float, dy_mm: float, feed_mm_min: float = 1500) -> None:
        with self._lock:
            self._check_not_alarmed()
            self.laser_off()
            self._send_line(f"$J=G91 G21 X{dx_mm:.3f} Y{dy_mm:.3f} F{feed_mm_min:.0f}")
            self._read_response()
            self._wait_until_idle()

    def _wait_until_idle(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_state()
            if state == "Idle":
                return
            if state == "Alarm":
                raise GrblAlarmError(
                    "Hit a limit switch mid-jog (ALARM) - click Unlock or "
                    "Home before jogging again."
                )
            time.sleep(0.1)
        raise GrblError("Timed out waiting for the jog to finish")

    def unlock(self) -> None:
        """Clears an ALARM (e.g. from a limit switch) in place. Position
        stays approximate until you run home() - this just lets you jog
        again immediately, matching LightBurn's lightning-bolt button.
        """
        with self._lock:
            self._send_line("$X")
            self._read_response()

    def home(self) -> None:
        """Runs GRBL's homing cycle: drives toward the limit switches and
        re-zeros against them, clearing any alarm and re-establishing a
        known position - matches LightBurn's house button. Can take a while
        and physically moves the machine to its home corner.
        """
        with self._lock:
            self._send_line("$H")
            self._read_response(timeout=HOMING_TIMEOUT_S)
            self._wait_until_idle(timeout=HOMING_TIMEOUT_S)
            self._refresh_wco()

    def set_origin_here(self) -> None:
        """Defines the current position as the new work zero (0,0) -
        matches LightBurn's 'set origin' function. Persists in GRBL until
        changed again or the controller is power-cycled; doesn't move
        anything, just relabels where 'home base' is for jogging/exports.
        """
        with self._lock:
            self._send_line("G10 L20 P1 X0 Y0")
            self._read_response()
            self._refresh_wco()

    def laser_off(self) -> None:
        with self._lock:
            self._send_line("M5")
            self._read_response()

    def close(self) -> None:
        with self._lock:
            try:
                self.laser_off()
            except GrblError:
                pass
            self._ser.close()
