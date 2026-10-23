"""Constantes du service BLE "rowing" du PM5.

Source : Concept2 "PM CSAFE Communication Definition", revision 0.31 (2025),
section Smart Bluetooth. UUID = base CE06XXXX-43E5-11E4-916C-0800200C9A66 avec
XXXX l'identifiant court de la caracteristique.
"""

from __future__ import annotations

from enum import IntEnum

ROWING_SERVICE = 0x0030
GENERAL_STATUS = 0x0031  # 19 octets
ADDITIONAL_STATUS_1 = 0x0032  # 17 octets
STATUS_SAMPLE_RATE = 0x0034  # ecriture : 0=1 s, 1=500 ms (defaut), 2=250 ms, 3=100 ms
STROKE_DATA = 0x0035  # 20 octets
ADDITIONAL_STROKE_DATA = 0x0036  # 18 octets
FORCE_CURVE = 0x003D  # 2 a 20 octets, plusieurs notifications par courbe (PM5v2 seulement)
MULTIPLEXED = 0x0080  # 1 octet d'identifiant + donnees, si les notifications directes sont off

# Notifications souscrites par l'enregistreur (en direct : pas de multiplexage).
RECORDED_CHARACTERISTICS = (
    GENERAL_STATUS, ADDITIONAL_STATUS_1, STROKE_DATA, ADDITIONAL_STROKE_DATA, FORCE_CURVE,
)

SAMPLE_RATE_100MS = 3
LBF_TO_N = 4.4482216152605


class StrokeState(IntEnum):
    WAITING_MIN_SPEED = 0
    WAITING_ACCELERATE = 1
    DRIVING = 2
    DWELLING = 3
    RECOVERY = 4


def char_uuid(short_id: int) -> str:
    """UUID complet (minuscules) d'une caracteristique C2 a partir de son id court."""
    return f"ce06{short_id:04x}-43e5-11e4-916c-0800200c9a66"
