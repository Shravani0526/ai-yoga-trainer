"""Hold timer that only runs while the posture is correct AND stable.

Behaviour
    * the pose must be correct for ``enter_s`` seconds before counting starts
    * counting pauses when the pose has been incorrect for longer than ``grace_s``
      (short tracking glitches do not break the hold)
    * time only accumulates while counting; paused time is never added
    * frame gaps larger than ``max_dt`` (video stalls) are clipped, so lag cannot inflate the timer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HoldState:
    held_s: float = 0.0           # total seconds of correct, stable holding
    streak_s: float = 0.0         # current uninterrupted hold
    best_streak_s: float = 0.0
    active: bool = False          # True while the timer is running
    breaks: int = 0               # how many times the hold was lost
    target_s: float = 20.0
    completed: bool = False

    @property
    def progress(self) -> float:
        return min(1.0, self.held_s / self.target_s) if self.target_s > 0 else 0.0


class HoldTimer:
    def __init__(self, target_s: float = 20.0, enter_s: float = 0.4, grace_s: float = 0.6,
                 max_dt: float = 0.25, strict: bool = False):
        self.target_s = float(target_s)
        self.enter_s = float(enter_s)
        self.grace_s = float(grace_s)
        self.max_dt = float(max_dt)
        self.strict = bool(strict)       # strict: the target must be reached in ONE streak
        self.reset()

    def reset(self) -> None:
        self.state = HoldState(target_s=self.target_s)
        self._last_t: Optional[float] = None
        self._good_run = 0.0
        self._bad_run = 0.0

    def set_target(self, target_s: float) -> None:
        self.target_s = float(target_s)
        self.state.target_s = self.target_s

    def update(self, t: float, is_correct: bool) -> HoldState:
        s = self.state
        if self._last_t is None:
            self._last_t = t
            return s
        dt = max(0.0, min(t - self._last_t, self.max_dt))
        self._last_t = t

        if is_correct:
            self._bad_run = 0.0
            self._good_run += dt
            if not s.active and self._good_run >= self.enter_s:
                s.active = True
            if s.active:
                s.held_s += dt
                s.streak_s += dt
                s.best_streak_s = max(s.best_streak_s, s.streak_s)
        else:
            self._good_run = 0.0
            self._bad_run += dt
            if s.active and self._bad_run > self.grace_s:
                s.active = False
                s.breaks += 1
                s.streak_s = 0.0

        reached = s.best_streak_s if self.strict else s.held_s
        if reached >= self.target_s:
            s.completed = True
        return s
