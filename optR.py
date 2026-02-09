"""
Optimized race telemetry extractor with:
1. Numba-accelerated acceleration calculations (2-5x speedup)
2. Numpy-vectorized laps_data processing
3. orjson for fast JSON serialization
4. Batch telemetry processing with single-pull-per-driver
5. Avoided DataFrame copies in acceleration path

Usage:
    # Run optimized version
    uv run python race_optimized.py

    # Compare with original
    uv run python benchmark_improvements.py
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
logger = logging.getLogger("race_optimized")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}

SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}

DRS_ACTIVE_VALUES = frozenset((10, 12, 14))


class TelemetryExtractorOptimized:
    """
    Optimized extractor for F1 race telemetry data.

    Optimizations:
    1. Numba JIT-compiled acceleration calculations
    2. Numpy-vectorized laps_data processing
    3. orjson for fast JSON serialization (~5-10x vs stdlib json)
    4. Single telemetry pull per driver, split via groupby
    5. Direct array-to-dict construction (no DataFrame copy for acc output)
    """

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
            # "São Paulo Grand Prix",
            # "Las Vegas Grand Prix",
            # "Qatar Grand Prix",
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

    def process_single_lap_telemetry(
        self, telemetry: pd.DataFrame, data_key: str
    ) -> Optional[Dict]:
        """Process telemetry for a single lap.

        Builds output dict directly from numpy arrays, avoiding a full
        DataFrame copy just to attach Ax/Ay/Az columns.
        """
        if telemetry.empty or len(telemetry) < 2:
            return None

        speed_vals = telemetry["Speed"].values
        time_vals = telemetry["Time"].values
        x_vals = telemetry["X"].values
        y_vals = telemetry["Y"].values
        z_vals = telemetry["Z"].values
        dist_vals = telemetry["Distance"].values

        vx_array = (speed_vals / 3.6).astype(np.float64)
        time_array = (time_vals / np.timedelta64(1, "s")).astype(np.float64)
        x_array = x_vals.astype(np.float64)
        y_array = y_vals.astype(np.float64)
        z_array = z_vals.astype(np.float64)
        dist_array = dist_vals.astype(np.float64)

        if self.use_numba:
            ax, ay, az = calculate_all_accelerations_numba(
                vx_array, time_array, x_array, y_array, z_array, dist_array, 3, 9, 9
            )
        else:
            ax, ay, az = self._calc_accelerations_numpy(
                vx_array, time_array, x_array, y_array, z_array, dist_array
            )

        drs_values = telemetry["DRS"].values
        drs_binary = np.isin(drs_values, list(DRS_ACTIVE_VALUES)).astype(np.int8)
        brake_binary = (telemetry["Brake"].values != 0).astype(np.int8)

        return {
            "tel": {
                "time": time_array.tolist(),
                "rpm": telemetry["RPM"].values.tolist(),
                "speed": speed_vals.tolist(),
                "gear": telemetry["nGear"].values.tolist(),
                "throttle": telemetry["Throttle"].values.tolist(),
                "brake": brake_binary.tolist(),
                "drs": drs_binary.tolist(),
                "distance": dist_vals.tolist(),
                "rel_distance": telemetry["RelativeDistance"].values.tolist(),
                "acc_x": ax.tolist(),
                "acc_y": ay.tolist(),
                "acc_z": az.tolist(),
                "x": x_vals.tolist(),
                "y": y_vals.tolist(),
                "z": z_vals.tolist(),
                "dataKey": data_key,
            }
        }

    @staticmethod
    def _calc_accelerations_numpy(
        vx_array: np.ndarray,
        time_array: np.ndarray,
        x_array: np.ndarray,
        y_array: np.ndarray,
        z_array: np.ndarray,
        dist_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numpy-based acceleration calculation (fallback)."""
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

        return ax, ay, az

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
        """Process laps and write individual telemetry files.

        Fetches telemetry once per driver, then splits by lap.
        """
        if not lap_numbers:
            return 0

        processed_count = 0

        try:
            pending_laps = []
            lap_file_paths = {}
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
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver).copy()

            telemetry_all = driver_laps.get_telemetry()
            if telemetry_all.empty or len(telemetry_all) < 2:
                return processed_count

            if "Distance" not in telemetry_all.columns:
                telemetry_all = telemetry_all.add_distance()
            if "RelativeDistance" not in telemetry_all.columns:
                telemetry_all = telemetry_all.add_relative_distance()

            if "LapNumber" not in telemetry_all.columns:
                logger.warning(
                    f"Telemetry missing LapNumber for {driver} in {event} {session}"
                )
                return processed_count

            required_columns = {
                "Speed",
                "Time",
                "X",
                "Y",
                "Z",
                "Distance",
                "RelativeDistance",
                "DRS",
                "Brake",
                "RPM",
                "nGear",
                "Throttle",
            }
            missing_columns = required_columns.difference(telemetry_all.columns)
            if missing_columns:
                logger.warning(
                    "Telemetry missing required columns for %s in %s %s: %s",
                    driver,
                    event,
                    session,
                    sorted(missing_columns),
                )
                return processed_count

            telemetry_all = telemetry_all.sort_values("Time").drop_duplicates(
                subset=["Time"]
            )
            telemetry_all = telemetry_all.reset_index(drop=True)
            telemetry_all = telemetry_all.dropna(subset=["LapNumber"])
            telemetry_all["LapNumberInt"] = telemetry_all["LapNumber"].astype(int)

            for lap_num in pending_laps:
                try:
                    telemetry = telemetry_all[
                        telemetry_all["LapNumberInt"] == int(lap_num)
                    ]
                    if telemetry.empty or len(telemetry) < 2:
                        continue

                    data_key = f"{self.year}-{event}-{session}-{driver}-{lap_num}"
                    telemetry_data = self.process_single_lap_telemetry(
                        telemetry, data_key
                    )
                    if telemetry_data is None:
                        continue

                    file_path = lap_file_paths[lap_num]
                    with open(file_path, "wb") as json_file:
                        json_file.write(orjson.dumps(telemetry_data))

                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing lap {lap_num} for {driver}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in batch processing for {driver}: {str(e)}")

        return processed_count

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver."""
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

    def get_circuit_info(self, event: str, session: str) -> Optional[Dict[str, List]]:
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
                url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
                response = requests.get(url, headers=HEADERS, timeout=10)
                if response.status_code == 200:
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
                        circuit_df = pd.DataFrame(
                            rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                        )
                        corner_info = {
                            "CornerNumber": circuit_df["Number"].tolist(),
                            "X": circuit_df["X"].tolist(),
                            "Y": circuit_df["Y"].tolist(),
                            "Angle": circuit_df["Angle"].tolist(),
                            "Distance": (circuit_df["Distance"] / 10).tolist(),
                            "Rotation": rotation,
                        }
                        CIRCUIT_INFO_CACHE[cache_key] = corner_info
                        return corner_info

            return None
        except Exception as e:
            logger.error(f"Error getting circuit info: {str(e)}")
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
            for entry in data["corners"]:
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

            return (
                pd.DataFrame(
                    rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                ),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None, 0.0

    def process_event_session(self, event: str, session: str) -> None:
        """Process a single event and session."""
        logger.info(f"Processing {event} - {session}")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            laps = f1session.laps
            drivers = list(laps["Driver"].unique())

            team_by_driver = (
                laps.drop_duplicates("Driver").set_index("Driver")["Team"].to_dict()
            )
            drivers_info = {
                "drivers": [
                    {"driver": d, "team": team_by_driver.get(d)}
                    for d in drivers
                ]
            }
            with open(f"{base_dir}/drivers.json", "wb") as json_file:
                json_file.write(orjson.dumps(drivers_info))

            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "wb") as json_file:
                    json_file.write(orjson.dumps(corner_info))

            with ThreadPoolExecutor(max_workers=min(4, len(drivers))) as executor:
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
        """Process all configured events and sessions."""
        logger.info(f"Starting optimized telemetry extraction for {self.year}")
        logger.info(f"Numba acceleration: {self.use_numba}")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        start_time = time.time()

        for event in self.events:
            for session in self.sessions:
                try:
                    self.process_event_session(event, session)
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing {event} {session}: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed_time:.2f} seconds")
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


def main():
    """Main entry point."""
    extractor = TelemetryExtractorOptimized(
        year=2025,
        events=["Abu Dhabi Grand Prix"],
        sessions=["Race"],
        use_numba=True,
    )

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
                f"Data not yet available. Waiting {wait_time} seconds before retry ({attempt}/{max_attempts})..."
            )
            time.sleep(wait_time)

    if attempt >= max_attempts:
        logger.error(
            f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
        )


if __name__ == "__main__":
    main()
