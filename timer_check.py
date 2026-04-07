# timer_check.py
# Manages per-vehicle timers for no-parking zone dwell time.
# Now includes stationarity gating: the timer only runs when the
# background subtractor confirms the vehicle is truly stationary.
# Also tracks per-zone violation statistics.

import time
import datetime


class ParkingTimer:
    """
    Tracks how long each vehicle has been stationary inside a no-parking zone
    and records violations when the dwell time exceeds the threshold.

    Upgrade: Timer is gated by a stationarity check — the vehicle must be
    confirmed as non-moving before the clock starts ticking.
    """

    def __init__(self, threshold_seconds=30):
        """
        Args:
            threshold_seconds (int): Dwell time limit before a vehicle is
                                     flagged as illegally parked.
        """
        self.threshold = threshold_seconds
        self._entry_times = {}       # { vehicle_id: entry_timestamp }
        self._zone_map = {}          # { vehicle_id: zone_index }
        self._stationary = {}        # { vehicle_id: bool }
        self.violations = {}         # { vehicle_id: violation_info_dict }
        self.zone_violation_counts = {}  # { zone_name: count }
        self.violation_timeline = []     # List of (timestamp_float, zone_name)

    def update_threshold(self, threshold_seconds):
        """Update the time threshold (e.g. when the user moves the slider)."""
        self.threshold = threshold_seconds

    # ------------------------------------------------------------------ #
    #  Stationarity gating                                                 #
    # ------------------------------------------------------------------ #

    def set_stationary(self, vehicle_id, is_stationary):
        """
        Update the stationarity status of a vehicle.

        Args:
            vehicle_id    (int):  Tracker-assigned vehicle ID.
            is_stationary (bool): True if vehicle is confirmed stationary by BGS.
        """
        self._stationary[vehicle_id] = is_stationary

    def is_vehicle_stationary(self, vehicle_id):
        """Return whether the vehicle is currently marked as stationary."""
        return self._stationary.get(vehicle_id, False)

    # ------------------------------------------------------------------ #
    #  Zone entry / exit                                                   #
    # ------------------------------------------------------------------ #

    def vehicle_in_zone(self, vehicle_id, zone_index, zone_name):
        """
        Call this every frame a vehicle is detected inside a zone.
        Only records the entry time on the first call for that vehicle,
        AND only if the vehicle is confirmed stationary.

        Args:
            vehicle_id  (int): Tracker-assigned vehicle ID.
            zone_index  (int): Index of the zone the vehicle is in.
            zone_name   (str): Human-readable zone name (e.g. "Zone 1").
        """
        # Only start the timer if the vehicle is stationary
        if not self._stationary.get(vehicle_id, False):
            return

        if vehicle_id not in self._entry_times:
            self._entry_times[vehicle_id] = time.time()
            self._zone_map[vehicle_id] = zone_index

    def vehicle_out_of_zone(self, vehicle_id):
        """
        Call this when a vehicle is no longer inside any zone.
        Clears the timer so the vehicle can be re-timed if it re-enters.

        Args:
            vehicle_id (int): Tracker-assigned vehicle ID.
        """
        self._entry_times.pop(vehicle_id, None)
        self._zone_map.pop(vehicle_id, None)
        self._stationary.pop(vehicle_id, None)

    # ------------------------------------------------------------------ #
    #  Duration and violation checks                                       #
    # ------------------------------------------------------------------ #

    def get_duration(self, vehicle_id):
        """
        Return how many seconds the vehicle has been in the zone.

        Returns:
            float: Elapsed seconds, or 0.0 if the vehicle is not tracked.
        """
        if vehicle_id in self._entry_times:
            return time.time() - self._entry_times[vehicle_id]
        return 0.0

    def is_violation(self, vehicle_id):
        """
        Return True if the vehicle has exceeded the time threshold.

        Returns:
            bool
        """
        return self.get_duration(vehicle_id) >= self.threshold

    def record_violation(self, vehicle_id, zone_name, vehicle_type, timestamp=None):
        """
        Log a violation for a vehicle (only once per vehicle ID).
        Updates the duration on subsequent calls.
        Also tracks per-zone counts and timeline.

        Args:
            vehicle_id   (int): Tracker ID.
            zone_name    (str): Zone where the violation occurred.
            vehicle_type (str): Class label (car / truck / bus / motorcycle).
            timestamp    (str): Human-readable time string; defaults to now.

        Returns:
            bool: True if this is a NEW violation entry, False if updated.
        """
        duration = self.get_duration(vehicle_id)
        ts = timestamp or datetime.datetime.now().strftime("%H:%M:%S")

        if vehicle_id not in self.violations:
            self.violations[vehicle_id] = {
                "vehicle_id":   vehicle_id,
                "vehicle_type": vehicle_type,
                "zone":         zone_name,
                "timestamp":    ts,
                "duration":     duration,
            }
            # Update per-zone stats
            self.zone_violation_counts[zone_name] = (
                self.zone_violation_counts.get(zone_name, 0) + 1
            )
            # Record on the timeline
            self.violation_timeline.append((time.time(), zone_name))

            return True  # Brand-new violation
        else:
            # Refresh duration for existing violation
            self.violations[vehicle_id]["duration"] = duration
            return False

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def get_all_violations(self):
        """Return a list of all recorded violation dicts."""
        return list(self.violations.values())

    def get_zone_stats(self):
        """Return a dict of { zone_name: violation_count }."""
        return dict(self.zone_violation_counts)

    def get_timeline(self):
        """Return the violation timeline as list of (timestamp_float, zone_name)."""
        return list(self.violation_timeline)

    def clear(self):
        """Reset all timers and violations (call on detection restart)."""
        self._entry_times.clear()
        self._zone_map.clear()
        self._stationary.clear()
        self.violations.clear()
        self.zone_violation_counts.clear()
        self.violation_timeline.clear()
