"""
Optimized race telemetry extractor with:
1. Numba-accelerated acceleration calculations (2-5x speedup)
2. orjson for fast JSON serialization
3. Vectorized lap data processing (numpy arrays instead of Python loops)
4. Bulk telemetry retrieval per driver (single fetch + split vs per-lap fetch)
5. Vectorized DRS/Brake binary conversion (no pandas .apply)
6. Explicit memory management (gc + session cache eviction)
7. File existence checks to skip already-processed laps

Usage:
    uv run python optR.py
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import orjson
import pandas as pd
import requests

import utils
from acceleration_numba import calculate_all_accelerations_numba, warm_up_jit

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

SESSION_CACHE: Dict = {}
CIRCUIT_INFO_CACHE: Dict = {}


class TelemetryExtractorOptimized:
    """Optimized class to handle extraction of F1 race telemetry data."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
        use_numba: bool = True,
    ):
        self.year = year
        self.use_numba = use_numba

        self.events = events or [
            # "Pre-Season Testing",
            # "Australian Grand Prix",
            # "Chinese Grand Prix",
            # "Japanese Grand Prix",
            # "Bahrain Grand Prix",
            # 'Saudi Arabian Grand Prix',
            # "Miami Grand Prix",
            # "Emilia Romagna Grand Prix",
            # "Monaco Grand Prix",
            # 'Spanish Grand Prix',
            # "Canadian Grand Prix",
            # "Austrian Grand Prix",
            # "British Grand Prix",
            # "Belgian Grand Prix",
            # "Hungarian Grand Prix",
            # "Dutch Grand Prix",
            # 'Italian Grand Prix',
            # 'Azerbaijan Grand Prix',
            # 'Singapore Grand Prix',
            # 'United States Grand Prix',
            # 'Mexico City Grand Prix',
            # 'São Paulo Grand Prix',
            # 'Las Vegas Grand Prix',
            # 'Qatar Grand Prix',
            "Abu Dhabi Grand Prix",
        ]
        self.sessions = sessions or ["Race"]

        if use_numba:
            logger.info("Warming up Numba JIT...")
            warm_up_jit()
            logger.info("Numba JIT ready")

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached session object to prevent reloading."""
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """Get list of driver codes for a given event and session."""
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Get drivers available for a given event and session."""
        try:
            f1session = self.get_session(event, session)
            laps = f1session.laps
            unique_drivers = laps["Driver"].unique()

            drivers = [
                {
                    "driver": driver,
                    "team": laps[laps.Driver == driver].Team.iloc[0],
                }
                for driver in unique_drivers
            ]

            return {"drivers": drivers}
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {str(e)}")
            return {"drivers": []}

    def laps_data(self, driver_laps: pd.DataFrame) -> Dict[str, List]:
        """Get lap data for a specific driver (numpy-vectorized)."""
        try:

            def timedelta_series_to_list(series: pd.Series) -> List:
                if series.empty:
                    return []
                arr = series.values
                mask = pd.isna(arr)
                result = np.empty(len(arr), dtype=object)
                valid = ~mask
                if valid.any():
                    valid_vals = arr[valid]
                    result[valid] = np.round(valid_vals / np.timedelta64(1, "s"), 3)
                result[mask] = "None"
                return result.tolist()

            def numeric_to_list(series: pd.Series, as_int: bool = False) -> List:
                if series.empty:
                    return []
                arr = series.values
                mask = pd.isna(arr)
                result = np.empty(len(arr), dtype=object)
                if as_int:
                    result[~mask] = arr[~mask].astype(int)
                else:
                    result[~mask] = arr[~mask]
                result[mask] = "None"
                return result.tolist()

            def string_to_list(series: pd.Series) -> List:
                if series.empty:
                    return []
                arr = series.values
                mask = pd.isna(arr)
                result = np.empty(len(arr), dtype=object)
                result[~mask] = arr[~mask].astype(str)
                result[mask] = "None"
                return result.tolist()

            def bool_to_list(series: pd.Series) -> List:
                if series.empty:
                    return []
                arr = series.values
                mask = pd.isna(arr)
                result = np.empty(len(arr), dtype=object)
                result[~mask] = arr[~mask].astype(bool)
                result[mask] = "None"
                return result.tolist()

            return {
                "time": timedelta_series_to_list(driver_laps["LapTime"]),
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": string_to_list(driver_laps["Compound"]),
                "stint": numeric_to_list(driver_laps["Stint"], as_int=True),
                "s1": timedelta_series_to_list(driver_laps["Sector1Time"]),
                "s2": timedelta_series_to_list(driver_laps["Sector2Time"]),
                "s3": timedelta_series_to_list(driver_laps["Sector3Time"]),
                "life": numeric_to_list(driver_laps["TyreLife"], as_int=True),
                "pos": numeric_to_list(driver_laps["Position"], as_int=True),
                "status": string_to_list(driver_laps["TrackStatus"]),
                "pb": bool_to_list(driver_laps["IsPersonalBest"]),
            }
        except Exception as e:
            logger.error(f"Error getting lap data: {str(e)}")
            return {
                "time": [],
                "lap": [],
                "compound": [],
                "stint": [],
                "s1": [],
                "s2": [],
                "s3": [],
                "life": [],
                "pos": [],
                "status": [],
                "pb": [],
            }

    def accCalc_numba(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Calculate acceleration using Numba-compiled functions."""
        vx_array = (telemetry["Speed"].values / 3.6).astype(np.float64)
        time_array = (telemetry["Time"].values / np.timedelta64(1, "s")).astype(
            np.float64
        )
        x_array = telemetry["X"].values.astype(np.float64)
        y_array = telemetry["Y"].values.astype(np.float64)
        z_array = telemetry["Z"].values.astype(np.float64)
        dist_array = telemetry["Distance"].values.astype(np.float64)

        ax, ay, az = calculate_all_accelerations_numba(
            vx_array, time_array, x_array, y_array, z_array, dist_array, 3, 9, 9
        )

        telemetry = telemetry.copy()
        telemetry["Ax"] = ax
        telemetry["Ay"] = ay
        telemetry["Az"] = az
        return telemetry

    def accCalc_numpy(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Original numpy-based acceleration calculation (fallback)."""
        vx_array = (telemetry["Speed"].values / 3.6).astype(np.float64)
        time_array = (telemetry["Time"].values / np.timedelta64(1, "s")).astype(
            np.float64
        )
        x_array = telemetry["X"].values.astype(np.float64)
        y_array = telemetry["Y"].values.astype(np.float64)
        z_array = telemetry["Z"].values.astype(np.float64)
        dist_array = telemetry["Distance"].values.astype(np.float64)

        dx = np.gradient(x_array)
        ds = np.gradient(dist_array)
        dtime = np.gradient(time_array)
        vx_sq = np.square(vx_array)

        kernel_3 = np.ones(3, dtype=np.float64) / 3
        kernel_9 = np.ones(9, dtype=np.float64) / 9

        ax = np.gradient(vx_array) / dtime
        ax = np.convolve(ax, kernel_3, mode="same")

        dy = np.gradient(y_array)
        theta = np.arctan2(dy, dx + np.finfo(float).eps)
        theta[0] = theta[1]
        theta_unwrap = np.unwrap(theta)
        dtheta = np.gradient(theta_unwrap)
        C = dtheta / (ds + 0.0001)
        ay = vx_sq * C
        ay[np.abs(ay) > 150] = 0
        ay = np.convolve(ay, kernel_9, mode="same")

        dz = np.gradient(z_array)
        z_theta = np.arctan2(dz, dx + np.finfo(float).eps)
        z_theta[0] = z_theta[1]
        z_theta_unwrap = np.unwrap(z_theta)
        z_dtheta = np.gradient(z_theta_unwrap)
        z_C = z_dtheta / (ds + 0.0001)
        az = vx_sq * z_C
        az[np.abs(az) > 150] = 0
        az = np.convolve(az, kernel_9, mode="same")

        telemetry = telemetry.copy()
        telemetry["Ax"] = ax
        telemetry["Ay"] = ay
        telemetry["Az"] = az
        return telemetry

    def process_single_lap_telemetry(
        self, telemetry: pd.DataFrame, data_key: str
    ) -> Dict:
        """Process telemetry for a single lap using vectorized operations."""
        if self.use_numba:
            acc_tel = self.accCalc_numba(telemetry)
        else:
            acc_tel = self.accCalc_numpy(telemetry)

        time_sec = acc_tel["Time"].dt.total_seconds().values
        drs_values = acc_tel["DRS"].values
        drs_binary = np.isin(drs_values, [10, 12, 14]).astype(np.int8)
        brake_binary = (acc_tel["Brake"].values != 0).astype(np.int8)

        return {
            "tel": {
                "time": time_sec.tolist(),
                "rpm": acc_tel["RPM"].values.tolist(),
                "speed": acc_tel["Speed"].values.tolist(),
                "gear": acc_tel["nGear"].values.tolist(),
                "throttle": acc_tel["Throttle"].values.tolist(),
                "brake": brake_binary.tolist(),
                "drs": drs_binary.tolist(),
                "distance": acc_tel["Distance"].values.tolist(),
                "rel_distance": acc_tel["RelativeDistance"].values.tolist(),
                "acc_x": acc_tel["Ax"].tolist(),
                "acc_y": acc_tel["Ay"].tolist(),
                "acc_z": acc_tel["Az"].tolist(),
                "x": acc_tel["X"].values.tolist(),
                "y": acc_tel["Y"].values.tolist(),
                "z": acc_tel["Z"].values.tolist(),
                "dataKey": data_key,
            }
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
        """Process laps using bulk telemetry retrieval and write individual files."""
        if not lap_numbers:
            return 0

        processed_count = 0

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            if driver_laps is None:
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver).copy()

            all_telemetry = driver_laps.get_telemetry()
            if all_telemetry.empty:
                return 0

            has_lap_number = "LapNumber" in all_telemetry.columns

            if has_lap_number:
                telemetry_by_lap = {
                    int(lap_num): group
                    for lap_num, group in all_telemetry.groupby("LapNumber", sort=False)
                }
            else:
                telemetry_by_lap = {}
                tel_time = all_telemetry["Time"].values
                lap_numbers_set = set(lap_numbers)
                for _, lap_row in driver_laps.iterrows():
                    lap_num = int(lap_row["LapNumber"])
                    if lap_num not in lap_numbers_set:
                        continue
                    try:
                        lap_start = lap_row["LapStartTime"]
                        lap_end = lap_start + lap_row["LapTime"]
                        if pd.isna(lap_start) or pd.isna(lap_end):
                            continue
                        start_ns = lap_start.value
                        end_ns = lap_end.value
                        mask = (tel_time >= start_ns) & (tel_time <= end_ns)
                        lap_tel = all_telemetry[mask]
                        if not lap_tel.empty:
                            telemetry_by_lap[lap_num] = lap_tel
                    except Exception:
                        pass

            for lap_num in lap_numbers:
                file_path = f"{driver_dir}/{lap_num}_tel.json"

                if os.path.exists(file_path):
                    processed_count += 1
                    continue

                try:
                    lap_tel = telemetry_by_lap.get(lap_num)
                    if lap_tel is None or lap_tel.empty:
                        continue

                    data_key = f"{self.year}-{event}-{session}-{driver}-{lap_num}"
                    telemetry_data = self.process_single_lap_telemetry(
                        lap_tel, data_key
                    )

                    with open(file_path, "wb") as json_file:
                        json_file.write(orjson.dumps(telemetry_data))

                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing lap {lap_num}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in batch processing for {driver}: {str(e)}")

        return processed_count

    def get_circuit_info(
        self, event: str, session: str
    ) -> Optional[Dict[str, List]]:
        """Get circuit corner information."""
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
                circuit_info, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_info = {
                        "CornerNumber": circuit_info["Number"].tolist(),
                        "X": circuit_info["X"].tolist(),
                        "Y": circuit_info["Y"].tolist(),
                        "Angle": circuit_info["Angle"].tolist(),
                        "Distance": (circuit_info["Distance"] / 10).tolist(),
                        "Rotation": rotation,
                    }
                    CIRCUIT_INFO_CACHE[cache_key] = corner_info
                    return corner_info

            logger.warning(f"Could not get corner data for {event} {session}")
            return None
        except Exception as e:
            logger.error(f"Error getting circuit info for {event} {session}: {str(e)}")
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """Get circuit information from the MultiViewer API."""
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

            rows = []
            for entry in data.get("corners", []):
                rows.append(
                    (
                        float(entry.get("trackPosition", {}).get("x", 0.0)),
                        float(entry.get("trackPosition", {}).get("y", 0.0)),
                        int(entry.get("number", 0)),
                        str(entry.get("letter", "")),
                        float(entry.get("angle", 0.0)),
                        float(entry.get("length", 0.0)),
                    )
                )

            if rows:
                return (
                    pd.DataFrame(
                        rows,
                        columns=["X", "Y", "Number", "Letter", "Angle", "Distance"],
                    ),
                    rotation,
                )
            return None, 0.0
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None, 0.0

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver using bulk telemetry retrieval."""
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            if driver_laps.empty:
                logger.warning(f"No laps for driver {driver}")
                return

            laptimes = self.laps_data(driver_laps)
            with open(f"{driver_dir}/laptimes.json", "wb") as json_file:
                json_file.write(orjson.dumps(laptimes))

            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            lap_numbers = driver_laps["LapNumber"].tolist()

            self.process_lap_batch(
                event, session, driver, lap_numbers, driver_dir, f1session, driver_laps
            )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        """Process a single event and session, extracting all telemetry data."""
        logger.info(f"Processing {event} - {session}")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            drivers_info = self.session_drivers(event, session)
            with open(f"{base_dir}/drivers.json", "wb") as json_file:
                json_file.write(orjson.dumps(drivers_info))

            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "wb") as json_file:
                    json_file.write(orjson.dumps(corner_info))

            drivers = self.session_drivers_list(event, session)

            max_workers = min(2, len(drivers))
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_driver,
                            event,
                            session,
                            driver,
                            base_dir,
                            f1session,
                        ): driver
                        for driver in drivers
                    }
                    for future in as_completed(futures):
                        driver = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error processing driver {driver}: {str(e)}")
            else:
                for driver in drivers:
                    self.process_driver(event, session, driver, base_dir, f1session)

            cache_key = f"{self.year}-{event}-{session}"
            SESSION_CACHE.pop(cache_key, None)
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self) -> None:
        """Process all configured events and sessions."""
        logger.info(
            f"Starting optimized telemetry extraction for {self.year} season"
        )
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")
        logger.info(f"Numba acceleration: {self.use_numba}")

        start_time = time.time()

        for event in self.events:
            for session in self.sessions:
                try:
                    self.process_event_session(event, session)
                    gc.collect()
                except Exception as e:
                    logger.error(
                        f"Error processing {event} - {session}: {str(e)}"
                    )

        elapsed_time = time.time() - start_time
        logger.info(f"Telemetry extraction completed in {elapsed_time:.2f} seconds")
        gc.collect()


def is_data_available(year: int, events: List[str], sessions: List[str]) -> bool:
    """Check if data is available for the specified year, events, and sessions."""
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


def check_memory_usage(threshold_percent: int = 80) -> bool:
    """Check if memory usage exceeds threshold and clear caches if needed."""
    try:
        import psutil
    except ImportError:
        return False

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Current memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(
            f"Memory usage exceeds {threshold_percent}% threshold, clearing caches"
        )
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()
        gc.collect()

        new_memory_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(
            f"New memory usage after clearing caches: {new_memory_percent:.2f}%"
        )
        return True

    return False


def main():
    """Main entry point."""
    try:
        extractor = TelemetryExtractorOptimized(use_numba=True)

        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        wait_time = 30
        max_attempts = 720
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} season data...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info(
                    f"Data is available for {extractor.year} season. Starting extraction..."
                )
                extractor.process_all_data()
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time} seconds before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)
                check_memory_usage()

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
