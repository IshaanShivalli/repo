"""
SUN / MOON MODULE
Provides texture paths and celestial-body direction vectors for the
day/night sky system used by main.py.

Time-of-day convention (matches main.py / Minecraft):
    0     – midnight
    6000  – noon
    12000 – sunset begins
    13000 – night begins
    18000 – midnight
    23000 – sunrise begins
    24000 – wraps back to 0

get_celestial(t) returns:
    sun_dir   : (x, y, z) unit vector pointing FROM the world TOWARD the sun
    moon_dir  : (x, y, z) unit vector pointing FROM the world TOWARD the moon
    moon_tex  : str – path to the current moon-phase texture
"""

from __future__ import annotations
import math

# ── Texture paths ─────────────────────────────────────────────────────────────

_ENV = "textures/assets/minecraft/textures/environment"
_CEL = f"{_ENV}/celestial"

SUN_TEX: str = f"{_CEL}/sun.png"

# Eight moon phases in Minecraft order:
#   0 = Full Moon  →  1 = Waning Gibbous  →  2 = Third Quarter (Last Quarter)
#   3 = Waning Crescent  →  4 = New Moon  →  5 = Waxing Crescent
#   6 = First Quarter  →  7 = Waxing Gibbous
MOON_TEXS: list[str] = [
    f"{_CEL}/moon/full_moon.png",        # phase 0
    f"{_CEL}/moon/waning_gibbous.png",   # phase 1
    f"{_CEL}/moon/third_quarter.png",    # phase 2
    f"{_CEL}/moon/waning_crescent.png",  # phase 3
    f"{_CEL}/moon/new_moon.png",         # phase 4
    f"{_CEL}/moon/waxing_crescent.png",  # phase 5
    f"{_CEL}/moon/first_quarter.png",    # phase 6
    f"{_CEL}/moon/waxing_gibbous.png",   # phase 7
]

# ── Moon phase tracking ───────────────────────────────────────────────────────
# One full lunar cycle = 8 Minecraft days.
# We track elapsed in-game days via a simple counter that increments each time
# time_of_day wraps.  Since main.py doesn't expose a day counter we derive it
# from the raw time value passed in (which is cumulative mod 24000 inside
# World.tick, but we receive the already-modded value).  We keep a module-level
# day counter that we bump whenever we detect a wrap.

_day_counter: int = 0
_last_t: float    = 0.0


def _update_day(t: float) -> None:
    """Detect day wrap and increment the day counter."""
    global _day_counter, _last_t
    if t < _last_t:          # time wrapped around 24000 → new day
        _day_counter += 1
    _last_t = t


def moon_phase() -> int:
    """Return current moon phase index 0-7."""
    return _day_counter % 8


# ── Celestial angle helpers ───────────────────────────────────────────────────

def _sun_angle(t: float) -> float:
    """
    Map time-of-day to the sun's angle (radians) around the East-West axis.

    Minecraft convention:
        t = 6000  → sun at zenith (angle = π/2, i.e. straight up)
        t = 0/24000 → sun at horizon (angle = 0 or π)
        t = 18000 → sun at nadir (angle = -π/2, i.e. straight down / midnight)

    We map t ∈ [0, 24000) → angle ∈ [0, 2π).
    At t=6000 the sun should be directly overhead → angle = π/2.
    """
    # Shift so that t=6000 maps to angle=π/2
    # angle = 2π * (t / 24000) + offset
    # At t=0 we want angle=−π/2 (below horizon, just before sunrise)
    # → offset = −π/2  → angle = 2π*(t/24000) − π/2
    return 2.0 * math.pi * (t / 24000.0) - math.pi * 0.5


def _dir_from_angle(angle: float) -> tuple[float, float, float]:
    """
    Convert a celestial angle to a world-space direction vector.

    The sun/moon orbit in the X-Y plane (East-West axis = X, Up = Y).
    The Z component is 0 so the orbit is a vertical circle facing the player.

        x = cos(angle)   (East-West)
        y = sin(angle)   (Up-Down)
        z = 0.0
    """
    return (math.cos(angle), math.sin(angle), 0.0)


# ── Public API ────────────────────────────────────────────────────────────────

def get_celestial(
    t: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """
    Return celestial body information for the given time-of-day value.

    Parameters
    ----------
    t : float
        Current time of day in the range [0, 24000).

    Returns
    -------
    sun_dir  : (x, y, z) – unit vector pointing FROM world origin TOWARD the sun.
    moon_dir : (x, y, z) – unit vector pointing FROM world origin TOWARD the moon.
    moon_tex : str        – file path of the current moon-phase texture.
    """
    _update_day(t)

    sun_angle  = _sun_angle(t)
    # Moon is always opposite the sun (π radians away)
    moon_angle = sun_angle + math.pi

    sun_dir  = _dir_from_angle(sun_angle)
    moon_dir = _dir_from_angle(moon_angle)
    moon_tex = MOON_TEXS[moon_phase()]

    return sun_dir, moon_dir, moon_tex
