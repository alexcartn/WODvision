"""Enregistreur Bluetooth du PM5 (extra optionnel `pm5`, bibliotheque `bleak`).

Ecrit un journal d'evenements bruts horodates (heure hote) : le decodage et la
synchronisation avec la video se font hors ligne. Le PM5 doit avoir le sans-fil
active (More Options > Turn Wireless ON) et n'etre connecte a aucune autre
application. Les services sont lus directement (pas de multiplexage) ; le taux
d'echantillonnage de l'etat est regle a 100 ms pour situer les transitions de
phase du coup (caracteristique 0x0034).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wodvision.pm5 import spec
from wodvision.pm5.session import write_event, write_event_header

logger = logging.getLogger(__name__)


async def record_events(
    client: Any,
    out_path: str | Path,
    duration_s: float | None = None,
    stop: asyncio.Event | None = None,
    device_name: str = "",
    clock: Callable[[], float] = time.time,
) -> int:
    """Souscrit aux notifications d'un client BLE deja connecte et journalise les evenements.

    `client` doit exposer `write_gatt_char`, `start_notify` et `stop_notify` (interface
    de `bleak.BleakClient`, ce qui permet de tester avec un faux client).

    Returns:
        Nombre d'evenements ecrits.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = stop or asyncio.Event()
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        write_event_header(fh, device_name)

        def make_handler(char_id: int):
            def handler(_sender: Any, data: bytearray) -> None:
                nonlocal count
                write_event(fh, clock(), char_id, bytes(data))
                count += 1
            return handler

        try:
            await client.write_gatt_char(
                spec.char_uuid(spec.STATUS_SAMPLE_RATE), bytes([spec.SAMPLE_RATE_100MS]),
                response=True,
            )
        except Exception as exc:  # reglage facultatif : le PM5 garde son taux par defaut
            logger.warning("taux d'echantillonnage non regle (%s)", exc)

        for char_id in spec.RECORDED_CHARACTERISTICS:
            try:
                await client.start_notify(spec.char_uuid(char_id), make_handler(char_id))
            except Exception as exc:
                logger.warning("caracteristique 0x%04X indisponible (%s)", char_id, exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=duration_s)
        except TimeoutError:
            pass
        for char_id in spec.RECORDED_CHARACTERISTICS:
            try:
                await client.stop_notify(spec.char_uuid(char_id))
            except Exception:
                pass
    return count


async def record(
    out_path: str | Path,
    duration_s: float | None = None,
    address: str | None = None,
    name_prefix: str = "PM5",
    scan_timeout_s: float = 20.0,
) -> int:
    """Cherche un PM5, s'y connecte et enregistre jusqu'a `duration_s` (ou Ctrl-C)."""
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as exc:  # pragma: no cover - depend de l'extra
        raise RuntimeError(
            "installer l'extra : pip install -e '.[pm5]' (bibliotheque bleak)"
        ) from exc

    if address:
        device = await BleakScanner.find_device_by_address(address, timeout=scan_timeout_s)
    else:
        device = await BleakScanner.find_device_by_filter(
            lambda d, _ad: (d.name or "").startswith(name_prefix), timeout=scan_timeout_s
        )
    if device is None:
        raise RuntimeError(
            "PM5 introuvable : sans-fil actif (More Options > Turn Wireless ON), PM5 pas deja "
            "connecte a une autre application ?"
        )
    logger.info("connexion a %s (%s)", device.name, device.address)
    async with BleakClient(device) as client:
        return await record_events(client, out_path, duration_s, device_name=device.name or "")
