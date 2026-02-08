"""
Optimized race telemetry extractor with:
1. Numba-accelerated acceleration calculations (2-5x speedup)
2. Numpy-vectorized laps_data processing
3. orjson for fast JSON serialization
4. Batch telemetry processing

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


class TelemetryExtractorOptimized:
    """
    Optimized extractor for F1 race telemetry data.

    Optimizations:
    1. Numba JIT-compiled acceleration calculations
    2. Numpy-vectorized laps_data processing
    3. orjson for fast JSON serialization
    4. Batch telemetry processing with grouped data
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
        """Original numpy-based acceleration calculation."""
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
        """Process telemetry for a single lap."""
        if self.use_numba:
            acc_tel = self.accCalc_numba(telemetry)
        else:
            acc_tel = self.accCalc_numpy(telemetry)

        time_sec = acc_tel["Time"].dt.total_seconds().values
        drs_values = acc_tel["DRS"].values
        drs_binary = np.where(
            (drs_values == 10) | (drs_values == 12) | (drs_values == 14), 1, 0
        )
        brake_values = acc_tel["Brake"].values
        brake_binary = np.where(brake_values == True, 1, 0)

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
                "acc_x": acc_tel["Ax"].values.tolist(),
                "acc_y": acc_tel["Ay"].values.tolist(),
                "acc_z": acc_tel["Az"].values.tolist(),
                "x": acc_tel["X"].values.tolist(),
                "y": acc_tel["Y"].values.tolist(),
                "z": acc_tel["Z"].values.tolist(),
                "dataKey": data_key,
            }
        }

    def process_lap(
        self,
        event: str,
        session: str,
        driver: str,
        lap_number: int,
        driver_dir: str,
        f1session=None,
        driver_laps=None,
    ) -> bool:
        """Process a single lap for a driver."""
        file_path = f"{driver_dir}/{lap_number}_tel.json"

        if os.path.exists(file_path):
            return True

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            if driver_laps is None:
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver).copy()

            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

            if selected_lap.empty:
                logger.warning(
                    f"No data for {driver} lap {lap_number} in {event} {session}"
                )
                return False

            telemetry = selected_lap.get_telemetry()

            if telemetry.empty:
                logger.warning(
                    f"No telemetry for {driver} lap {lap_number} in {event} {session}"
                )
                return False

            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"
            telemetry_data = self.process_single_lap_telemetry(telemetry, data_key)

            with open(file_path, "wb") as json_file:
                json_file.write(orjson.dumps(telemetry_data))

            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return False

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

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        self.process_lap,
                        event,
                        session,
                        driver,
                        lap_number,
                        driver_dir,
                        f1session,
                        driver_laps,
                    )
                    for lap_number in lap_numbers
                ]

                for future in as_completed(futures):
                    future.result()

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

            team_colors = utils.team_colors(self.year)
            laps_with_color = laps.copy()
            laps_with_color["color"] = laps_with_color["Team"].map(team_colors)

            drivers_info = {
                "drivers": [
                    {"driver": d, "team": laps.loc[laps.Driver == d, "Team"].iloc[0]}
                    for d in drivers
                ]
            }
            with open(f"{base_dir}/drivers.json", "wb") as json_file:
                json_file.write(orjson.dumps(drivers_info))

            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "wb") as json_file:
                    json_file.write(orjson.dumps(corner_info))

            max_workers = min(2, len(drivers))
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            else:
                for driver in drivers:
                    self.process_driver(event, session, driver, base_dir, f1session)

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
