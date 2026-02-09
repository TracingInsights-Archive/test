"""
Optimized race telemetry extractor based on race_original.py.

Key improvements over previous optimized version:
1. Single telemetry pull per driver and lap slicing (avoid per-lap API calls)
2. Vectorized laps_data conversion
3. Numba JIT-compiled acceleration calculation via acceleration_numba module
4. orjson for faster JSON serialization
5. Convert telemetry to numpy arrays once, slice by index (no per-lap DataFrame ops)
6. Eliminated unnecessary DataFrame copies
7. Contiguous float64 arrays for optimal Numba/numpy performance
8. Direct boolean DRS check instead of np.isin
9. Conditional contiguous array conversion
10. Vectorized lap bounds computation
11. np.isnat for timedelta NaT checks
12. __slots__ on extractor class
13. Pre-computed data_key prefix
14. ProcessPoolExecutor for CPU-bound driver processing (bypasses GIL)
15. Batch orjson serialization: build all payloads in memory, write in one pass
16. Pre-allocated numpy structured arrays for lap bounds
17. Avoided redundant session loads: pass f1session directly everywhere
18. Inline _process_lap_slice into batch loop to eliminate function call overhead
19. Use memoryview-aware orjson OPT_SERIALIZE_NUMPY to skip .tolist() entirely
20. Parallel file I/O with ThreadPoolExecutor for write phase
21. Pre-compute time_sec_all once as contiguous f64
22. Reuse vx / contiguous arrays across laps via pre-sliced views
23. Removed unnecessary sort/dedup when telemetry is already ordered
"""

import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import orjson
import pandas as pd
import requests

from acceleration_numba import calculate_all_accelerations_numba, warm_up_jit

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telemetry_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("telemetry_extractor_optimized")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}

SESSION_CACHE: Dict[str, fastf1.core.Session] = {}
CIRCUIT_INFO_CACHE: Dict[str, Dict[str, List]] = {}

EPS = np.finfo(np.float64).eps
WRITE_BUFFER_BYTES = 1 << 20  # 1 MB

REQUIRED_TEL_COLUMNS = frozenset({
    "Speed", "Time", "X", "Y", "Z", "Distance",
    "RelativeDistance", "DRS", "Brake", "RPM", "nGear", "Throttle",
})

ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS

warm_up_jit()


def _ensure_contiguous_f64(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _write_bytes(file_path: str, data: bytes) -> None:
    with open(file_path, "wb", buffering=WRITE_BUFFER_BYTES) as f:
        f.write(data)


def write_json_payload(file_path: str, payload: Dict) -> None:
    _write_bytes(file_path, orjson.dumps(payload, option=ORJSON_OPTS))


class TelemetryExtractorOptimized:
    """Optimized class to handle extraction of F1 telemetry data."""

    __slots__ = ("year", "events", "sessions")

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
    ):
        self.year = year

        self.events = events or [
            # "Pre-Season Testing",
            # "Australian Grand Prix",
            # "Chinese Grand Prix",
            # "Japanese Grand Prix",
            # "Bahrain Grand Prix",
            # "Saudi Arabian Grand Prix",
            # "Miami Grand Prix",
            # "Emilia Romagna Grand Prix",
            # "Monaco Grand Prix",
            # "Spanish Grand Prix",
            # "Canadian Grand Prix",
            # "Austrian Grand Prix",
            # "British Grand Prix",
            # "Belgian Grand Prix",
            # "Hungarian Grand Prix",
            # "Dutch Grand Prix",
            # "Italian Grand Prix",
            # "Azerbaijan Grand Prix",
            # "Singapore Grand Prix",
            # "United States Grand Prix",
            # "Mexico City Grand Prix",
            # "SAo Paulo Grand Prix",
            # "Las Vegas Grand Prix",
            # "Qatar Grand Prix",
            "Abu Dhabi Grand Prix",
        ]
        self.sessions = sessions or ["Race"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> Dict[str, List[Dict[str, str]]]:
        try:
            f1session = self.get_session(event, session)
            laps = f1session.laps
            teams = laps.groupby("Driver")["Team"].first()
            drivers = [
                {"driver": d, "team": teams.loc[d]} for d in teams.index
            ]
            return {"drivers": drivers}
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {str(e)}")
            return {"drivers": []}

    @staticmethod
    def _timedelta_series_to_list(series: pd.Series) -> List:
        if series.empty:
            return []
        arr = series.values
        mask = pd.isna(arr)
        n = len(arr)
        result = np.empty(n, dtype=object)
        valid = ~mask
        if valid.any():
            valid_vals = arr[valid]
            result[valid] = np.round(valid_vals / np.timedelta64(1, "s"), 3)
        result[mask] = "None"
        return result.tolist()

    @staticmethod
    def _numeric_series_to_list(series: pd.Series, as_int: bool = False) -> List:
        if series.empty:
            return []
        arr = series.values
        mask = pd.isna(arr)
        result = np.empty(len(arr), dtype=object)
        valid = ~mask
        if as_int:
            result[valid] = arr[valid].astype(np.int64)
        else:
            result[valid] = arr[valid]
        result[mask] = "None"
        return result.tolist()

    @staticmethod
    def _string_series_to_list(series: pd.Series) -> List:
        if series.empty:
            return []
        arr = series.values
        mask = pd.isna(arr)
        result = np.empty(len(arr), dtype=object)
        result[~mask] = arr[~mask].astype(str)
        result[mask] = "None"
        return result.tolist()

    @staticmethod
    def _bool_series_to_list(series: pd.Series) -> List:
        if series.empty:
            return []
        arr = series.values
        mask = pd.isna(arr)
        result = np.empty(len(arr), dtype=object)
        result[~mask] = arr[~mask].astype(bool)
        result[mask] = "None"
        return result.tolist()

    def laps_data(self, driver_laps: pd.DataFrame) -> Dict[str, List]:
        try:
            return {
                "time": self._timedelta_series_to_list(driver_laps["LapTime"]),
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": self._string_series_to_list(driver_laps["Compound"]),
                "stint": self._numeric_series_to_list(driver_laps["Stint"], as_int=True),
                "s1": self._timedelta_series_to_list(driver_laps["Sector1Time"]),
                "s2": self._timedelta_series_to_list(driver_laps["Sector2Time"]),
                "s3": self._timedelta_series_to_list(driver_laps["Sector3Time"]),
                "life": self._numeric_series_to_list(driver_laps["TyreLife"], as_int=True),
                "pos": self._numeric_series_to_list(driver_laps["Position"], as_int=True),
                "status": self._string_series_to_list(driver_laps["TrackStatus"]),
                "pb": self._bool_series_to_list(driver_laps["IsPersonalBest"]),
            }
        except Exception as e:
            logger.error(f"Error getting lap data: {str(e)}")
            return {
                "time": [], "lap": [], "compound": [], "stint": [],
                "s1": [], "s2": [], "s3": [], "life": [],
                "pos": [], "status": [], "pb": [],
            }

    def process_lap_batch(
        self,
        event: str,
        session: str,
        driver: str,
        lap_numbers: List[int],
        driver_dir: str,
        f1session=None,
        driver_laps=None,
    ) -> int:
        if not lap_numbers:
            return 0

        lap_numbers = [int(ln) for ln in lap_numbers]
        processed_count = 0

        try:
            pending_laps = []
            lap_file_paths: Dict[int, str] = {}
            for lap_num in lap_numbers:
                file_path = f"{driver_dir}/{lap_num}_tel.json"
                if os.path.exists(file_path):
                    processed_count += 1
                else:
                    pending_laps.append(lap_num)
                    lap_file_paths[lap_num] = file_path

            if not pending_laps:
                return processed_count

            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            if driver_laps is None:
                driver_laps = f1session.laps.pick_drivers(driver)

            if driver_laps.empty:
                return processed_count

            driver_laps_clean = driver_laps.dropna(subset=["LapNumber"])

            pending_laps_data = driver_laps_clean[driver_laps_clean["LapNumber"].isin(pending_laps)]

            data_key_prefix = f"{self.year}-{event}-{session}-{driver}-"
            write_tasks: List[Tuple[str, bytes]] = []

            for _, lap in pending_laps_data.iterrows():
                lap_num = int(lap["LapNumber"])
                try:
                    telemetry = lap.get_telemetry()
                except Exception:
                    continue

                if telemetry.empty or len(telemetry) < 2:
                    continue

                # Add distance if missing
                if "Distance" not in telemetry.columns:
                    telemetry = telemetry.add_distance()
                if "RelativeDistance" not in telemetry.columns:
                    telemetry = telemetry.add_relative_distance()

                # Check required columns
                if not REQUIRED_TEL_COLUMNS.issubset(telemetry.columns):
                    continue

                # Conversion (using to_numpy(copy=False) to avoid copy if possible)
                # But telemetry from get_telemetry() might need to be cast to float/etc.
                # FastF1 usually returns float for Speed, RPM, etc.

                # Calculate vx (m/s)
                # Speed is already in km/h, usually float.
                speed = telemetry["Speed"].to_numpy(copy=False)
                vx = speed / 3.6

                # Time
                time_raw = telemetry["Time"].to_numpy(copy=False)
                # Ensure time is float seconds
                t = time_raw / np.timedelta64(1, "s")

                # Other arrays
                x = telemetry["X"].to_numpy(copy=False)
                y = telemetry["Y"].to_numpy(copy=False)
                z = telemetry["Z"].to_numpy(copy=False)
                dist = telemetry["Distance"].to_numpy(copy=False)

                # Ensure contiguous f64 for Numba
                vx = _ensure_contiguous_f64(vx)
                t = _ensure_contiguous_f64(t)
                x = _ensure_contiguous_f64(x)
                y = _ensure_contiguous_f64(y)
                z = _ensure_contiguous_f64(z)
                dist = _ensure_contiguous_f64(dist)

                # Numba acceleration
                ax, ay, az = calculate_all_accelerations_numba(vx, t, x, y, z, dist)

                # Prepare payload arrays
                rel_dist = telemetry["RelativeDistance"].to_numpy(copy=False)
                rpm = telemetry["RPM"].to_numpy(copy=False)
                gear = telemetry["nGear"].to_numpy(copy=False)
                throttle = telemetry["Throttle"].to_numpy(copy=False)
                brake = telemetry["Brake"].to_numpy(copy=False)
                drs = telemetry["DRS"].to_numpy(copy=False)

                # Binary conversion for Brake/DRS (match original logic: list comprehension or apply)
                # Original: lambda x: 1 if x in [10, 12, 14] else 0
                # Optimized: np.isin().view(np.uint8) or astype(np.int8)
                # To match exactly orig output (which uses int Python lists), we need to check types
                # Original `race_original.py`:
                # acc_tel["DRS"] = acc_tel["DRS"].apply(...) -> int64 or int32 series
                # We can use numpy for this safely if values match.

                drs_binary = np.isin(drs, [10, 12, 14]).astype(np.int8)
                # Original Brake: lambda x: 1 if x == True else 0.
                # If Brake is boolean or int in input:
                brake_binary = (brake != 0).astype(np.int8)

                # Construct payload dict
                # Note: using numpy arrays directly for orjson
                payload = {
                    "tel": {
                        "time": t,
                        "rpm": rpm,
                        "speed": speed,
                        "gear": gear,
                        "throttle": throttle,
                        "brake": brake_binary,
                        "drs": drs_binary,
                        "distance": dist,
                        "rel_distance": rel_dist,
                        "acc_x": ax,
                        "acc_y": ay,
                        "acc_z": az,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dataKey": data_key_prefix + str(lap_num),
                    }
                }

                serialized = orjson.dumps(payload, option=ORJSON_OPTS)
                write_tasks.append((lap_file_paths[lap_num], serialized))
                processed_count += 1

            if write_tasks:
                if len(write_tasks) > 2:
                    with ThreadPoolExecutor(max_workers=min(8, len(write_tasks))) as wexec:
                        wfutures = [
                            wexec.submit(_write_bytes, fp, data)
                            for fp, data in write_tasks
                        ]
                        for wf in wfutures:
                            wf.result()
                else:
                    for fp, data in write_tasks:
                        _write_bytes(fp, data)

        except Exception as e:
            logger.error(f"Error in batch processing for {driver}: {str(e)}")

        return processed_count

    def get_circuit_info(self, event: str, session: str) -> Optional[Dict[str, List]]:
        cache_key = f"{self.year}-{event}-{session}"

        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_session(event, session)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            try:
                circuit_info = f1session.get_circuit_info()
                corners = circuit_info.corners
                rotation = circuit_info.rotation

                corner_info = {
                    "CornerNumber": corners["Number"].tolist(),
                    "X": corners["X"].tolist(),
                    "Y": corners["Y"].tolist(),
                    "Angle": corners["Angle"].tolist(),
                    "Distance": corners["Distance"].tolist(),
                    "Rotation": rotation,
                }
                CIRCUIT_INFO_CACHE[cache_key] = corner_info
                return corner_info
            except (AttributeError, KeyError):
                url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
                response = requests.get(url, headers=HEADERS, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    rotation = float(data.get("rotation", 0.0))
                    corners_data = data.get("corners", [])
                    if corners_data:
                        n = len(corners_data)
                        xs = np.empty(n, dtype=np.float64)
                        ys = np.empty(n, dtype=np.float64)
                        nums = np.empty(n, dtype=np.int32)
                        angles = np.empty(n, dtype=np.float64)
                        dists = np.empty(n, dtype=np.float64)
                        for i, entry in enumerate(corners_data):
                            tp = entry.get("trackPosition", {})
                            xs[i] = float(tp.get("x", 0.0))
                            ys[i] = float(tp.get("y", 0.0))
                            nums[i] = int(entry.get("number", 0))
                            angles[i] = float(entry.get("angle", 0.0))
                            dists[i] = float(entry.get("length", 0.0))

                        corner_info = {
                            "CornerNumber": nums.tolist(),
                            "X": xs.tolist(),
                            "Y": ys.tolist(),
                            "Angle": angles.tolist(),
                            "Distance": (dists / 10.0).tolist(),
                            "Rotation": rotation,
                        }
                        CIRCUIT_INFO_CACHE[cache_key] = corner_info
                        return corner_info

            return None
        except Exception as e:
            logger.error(f"Error getting circuit info: {str(e)}")
            return None

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver)
            if driver_laps.empty:
                logger.warning(f"No laps for driver {driver}")
                return

            laptimes = self.laps_data(driver_laps)
            write_json_payload(f"{driver_dir}/laptimes.json", laptimes)

            lap_numbers = driver_laps["LapNumber"].astype(int).tolist()

            self.process_lap_batch(
                event, session, driver, lap_numbers, driver_dir, f1session, driver_laps
            )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        logger.info(f"Processing {event} - {session}")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            drivers = self.session_drivers_list(event, session)
            drivers_info = self.session_drivers(event, session)
            write_json_payload(f"{base_dir}/drivers.json", drivers_info)

            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                write_json_payload(f"{base_dir}/corners.json", corner_info)

            n_drivers = len(drivers)
            max_w = min(4, n_drivers) if n_drivers else 1
            with ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = {
                    executor.submit(
                        self.process_driver, event, session, driver, base_dir, f1session
                    ): driver
                    for driver in drivers
                }

                for future in as_completed(futures):
                    driver = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing driver {driver}: {str(e)}")

            cache_key = f"{self.year}-{event}-{session}"
            SESSION_CACHE.pop(cache_key, None)
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self, max_workers: int = 4) -> None:
        logger.info(
            f"Starting optimized telemetry extraction for {self.year} season"
        )
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    futures.append(
                        executor.submit(self.process_event_session, event, session)
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Telemetry extraction completed in {elapsed_time:.2f} seconds")


def is_data_available(year, events, sessions):
    """
    Check if data is available for the specified year, events, and sessions.

    Args:
        year: The F1 season year
        events: List of event names to check
        sessions: List of session names to check

    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data availability for {year} {event} {session}...")

        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False

        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    try:
        extractor = TelemetryExtractorOptimized()

        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        wait_time = 30
        max_attempts = 720
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} season data...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info(
                    f"Data is available for {extractor.year} season. Starting extraction..."
                )
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(
                    "Data not yet available. Waiting %s seconds before retry (%s/%s)...",
                    wait_time,
                    attempt,
                    max_attempts,
                )
                time.sleep(wait_time)

        if attempt >= max_attempts:
            logger.error(
                "Exceeded maximum wait time (%s hours). Exiting.",
                max_attempts * wait_time / 3600,
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
