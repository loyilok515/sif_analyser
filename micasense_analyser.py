"""
micasense_analyzer.py
─────────────────────
MicaSense RedEdge live capture and plant-health analysis, encapsulated
as a reusable class (ready for ROS2 node wrapping later).

Band map (1-indexed, matches MicaSense API):
    "1"  Green   528 nm
    "2"  Green2  570 nm
    "3"  Red     645 nm
    "4"  RedEdge 680 nm
    "5"  NIR     900 nm

Index / display band requirements
    RGB display  : 1 (G), 2 (G2), 3 (R)          ← pseudo-colour
    NDVI         : 3 (R),               5 (NIR)
    PRI          : 1 (530 nm proxy),    2 (570 nm) → (B1-B2)/(B1+B2)

Attitude-aware geotransform
    Roll, pitch, yaw (DLS IMU) are used to:
      • rotate the pixel-to-GPS mapping by heading (yaw)
      • record tilt metadata (roll, pitch) — severe tilt is flagged
"""

import io
import math
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import requests
import urllib3
from PIL import Image
import tifffile
import serial
from pathlib import Path


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Band map ──────────────────────────────────────────────────────────────────
BAND_MAP = {
    "1": ("Green",    528),
    "2": ("Green2",   570),
    "3": ("Red",      645),
    "4": ("RedEdge",  680),
    "5": ("NIR",      900),
}

# ── Band requirements per feature ────────────────────────────────────────────
_BANDS_FOR: dict[str, set[str]] = {
    "rgb":  {"1", "2", "3"},   # pseudo-colour (G, G2, R)
    "ndvi": {"3", "5"},
    "pri":  {"1", "2"},
}


def _required_bands(index_mode: str, want_rgb: bool = True) -> set[str]:
    needed: set[str] = set()
    if want_rgb:
        needed |= _BANDS_FOR["rgb"]
    for mode in ("ndvi", "pri"):
        if index_mode in (mode, "both"):
            needed |= _BANDS_FOR[mode]
    return needed


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    """2-D rotation matrix (CCW positive, acts in the East-North plane)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s],
                     [s,  c]])


class MicaSenseAnalyzer:
    """
    Encapsulates the full MicaSense RedEdge capture and analysis pipeline.

    Parameters
    ----------
    camera_ip            : camera base URL, e.g. "http://192.168.10.254"
    cache_format         : "jpeg" (8-bit, fast) or "tif" (16-bit raw)
    index_mode           : "ndvi", "pri", or "both"

    ndvi_stress_min/max  : NDVI stressed-vegetation range
    pri_stress_min/max   : PRI stressed-vegetation range
                           (PRI of healthy vegetation ≈ 0.05–0.15;
                            stressed / low-light ≈ −0.2–0.05)

    min_cluster_radius_m : clusters whose radius (in metres) exceeds this
                           value are flagged as "UNHEALTHY"
    n_clusters           : number of K-means clusters to form (e.g. 10)
    n_output_clusters    : max clusters to keep/transmit (the lowest-mean ones)
    min_pixels           : discard clusters smaller than this
    sensor_width_mm      : MicaSense sensor width  (default 4.8 mm)
    focal_length_mm      : lens focal length        (default 5.5 mm)
    ecc_iterations       : max ECC alignment iterations
    ecc_epsilon          : ECC convergence epsilon
    crop_fraction        : fraction trimmed from each edge before alignment
    want_rgb             : download bands needed for the RGB preview
    max_tilt_deg         : warn if |roll| or |pitch| exceeds this (degrees)
    tree_height          : estimated tree height for GSD calculations
    """

    BAND_LABELS = {k: f"Band {k}  {v[0]} {v[1]} nm" for k, v in BAND_MAP.items()}

    SEVERITY_COLORS = {
        "CRITICAL": (1.0, 0.0, 0.0),
        "SEVERE":   (1.0, 0.4, 0.0),
        "MODERATE": (1.0, 0.7, 0.0),
    }

    def __init__(
        self,
        camera_ip              = "http://192.168.10.254",
        cache_format           = "jpeg",
        index_mode             = "both",
        # ── per-index stress thresholds ──────────────────────────────
        ndvi_stress_min        = 0.10,
        ndvi_stress_max        = 0.35,
        pri_stress_min         = -0.20,   # PRI < 0.05 → stress / low xanthophyll
        pri_stress_max         =  0.05,
        # ── cluster health gate ───────────────────────────────────────
        min_cluster_radius_m   = 5.0,     # clusters larger than this → UNHEALTHY
        # ────────────────────────────────────────────────────────────
        n_clusters             = 6,
        n_output_clusters      = 2,
        min_pixels             = 30,
        sensor_width_mm        = 4.8,
        focal_length_mm        = 5.5,
        ecc_iterations         = 200,
        ecc_epsilon            = 1e-5,
        crop_fraction          = 0.25,
        want_rgb               = True,
        max_tilt_deg           = 15.0,
        tree_height            = 0.0,
        home_alt               = 0.0, 
        serial_port            = "/dev/ttyUSB0",
        serial_baud            = 57600,
        serial_settle_s        = 1.0,
        image_path             = None,
    ):
        self.camera_ip             = camera_ip.rstrip("/")
        self.cache_format          = cache_format.lower()
        self.index_mode            = index_mode.lower()
        self.min_cluster_radius_m  = min_cluster_radius_m
        self.max_tilt_deg          = max_tilt_deg
        self.tree_height           = tree_height
        self.home_alt              = home_alt

        self.stress_thresholds = {
            "NDVI": (ndvi_stress_min, ndvi_stress_max),
            "PRI":  (pri_stress_min,  pri_stress_max),
        }

        self.n_clusters      = n_clusters
        self.n_output_clusters = n_output_clusters
        self.min_pixels      = min_pixels
        self.sensor_width_mm = sensor_width_mm
        self.focal_length_mm = focal_length_mm
        self.ecc_iterations  = ecc_iterations
        self.ecc_epsilon     = ecc_epsilon
        self.crop_fraction   = crop_fraction
        self.want_rgb        = want_rgb
        self.serial_port     = serial_port
        self.serial_baud     = serial_baud
        self.serial_settle_s = serial_settle_s
        self.image_path      = image_path

        self._needed_bands: set[str] = _required_bands(
            self.index_mode, want_rgb=self.want_rgb
        )
        print(f"  Bands that will be downloaded: "
              f"{sorted(self._needed_bands)}  ({len(self._needed_bands)} of 5)")

        self.session = requests.Session()
        self.session.verify = False

        # State populated by run()
        self.last_bands_crop = None
        self.last_rgb        = None
        self.last_index_maps = None
        self.last_clusters   = None
        self.last_geo        = None
        self.last_attitude   = None

    # ──────────────────────────────────────────────────────────────────────────
    #  GPS  +  ATTITUDE
    # ──────────────────────────────────────────────────────────────────────────

    def obtain_gps(self) -> tuple[float, float, float]:
        """Return (lat_deg, lon_deg, alt_m) from /gps."""
        gps = self.session.get(f"{self.camera_ip}/gps", timeout=10).json()

        # /gps returns latitude / longitude already in radians on RedEdge
        lat_deg = math.degrees(gps["latitude"])
        lon_deg = math.degrees(gps["longitude"])
        alt_m   = gps["altitude"]

        fix_ok  = gps.get("fix3d", False)
        utc_ok  = gps.get("utc_time_valid", False)
        h_acc   = gps.get("p_acc", float("nan"))
        print(f"  GPS  fix3d={fix_ok}  lat={lat_deg:.7f}  lon={lon_deg:.7f}  "
              f"alt={alt_m:.1f} m  h_acc={h_acc:.2f} m  utc_valid={utc_ok}")

        if not fix_ok:
            print("  ⚠  WARNING: no 3-D GPS fix — location data may be invalid.")

        return lat_deg, lon_deg, alt_m

    def obtain_attitude(self) -> dict[str, float]:
        """
        Return attitude dict with roll, pitch, yaw in **degrees** from /dls_imu.

        The DLS IMU reports in radians; we convert here so the rest of the
        pipeline works in degrees throughout.
        """
        imu = self.session.get(f"{self.camera_ip}/dls_imu", timeout=10).json()

        roll_deg  = math.degrees(imu["roll"])
        pitch_deg = math.degrees(imu["pitch"])
        yaw_deg   = math.degrees(imu["yaw"])   # heading (0° = North, CW positive)

        print(f"  IMU  roll={roll_deg:+.3f}°  pitch={pitch_deg:+.3f}°  "
              f"yaw(heading)={yaw_deg:.3f}°")

        tilt = max(abs(roll_deg), abs(pitch_deg))
        if tilt > self.max_tilt_deg:
            print(f"  ⚠  WARNING: tilt {tilt:.1f}° exceeds limit "
                  f"{self.max_tilt_deg}°  — geotransform accuracy reduced.")

        return {
            "roll_deg":  roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg":   yaw_deg,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  CAPTURE — selective band download
    # ──────────────────────────────────────────────────────────────────────────

    def trigger_capture(self) -> dict[str, np.ndarray]:
        """
        Trigger one capture and download only the bands in self._needed_bands.

        Returns
        -------
        dict {"1": ndarray, "3": ndarray, …}  — float64, full-resolution
        """
        use_jpeg = self.cache_format == "jpeg"
        payload  = {
            "cache_jpeg":    31 if use_jpeg     else 0,
            "cache_raw":     31 if not use_jpeg else 0,
            "block":         True,
            "store_capture": True,
        }

        print(f"  Triggering capture (format={self.cache_format})…")
        r    = self.session.post(f"{self.camera_ip}/capture",
                                 json=payload, timeout=30)
        data = r.json()

        if data.get("status") != "complete":
            raise RuntimeError(f"Capture failed: {data}")
        print(f"  Capture complete at {data.get('time', '?')}")

        cache_key   = "jpeg_cache_path" if use_jpeg else "raw_cache_path"
        cache_paths = data.get(cache_key, {})
        if not cache_paths:
            raise RuntimeError(f"No '{cache_key}' in response.\n{data}")

        bands: dict[str, np.ndarray] = {}

        for band_key, path in sorted(cache_paths.items()):
            if band_key not in self._needed_bands:
                print(f"  Band {band_key} ({self.BAND_LABELS.get(band_key,'?')}): "
                      "skipped (not required)")
                continue

            resp = self.session.get(f"{self.camera_ip}{path}", timeout=15)

            if use_jpeg:
                img = Image.open(io.BytesIO(resp.content)).convert("L")
                arr = np.array(img, dtype=float)
            else:
                arr = tifffile.imread(io.BytesIO(resp.content)).astype(float)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]

            bands[band_key] = arr
            print(f"  Band {band_key} ({self.BAND_LABELS.get(band_key,'?')}): "
                  f"{arr.shape}  min={arr.min():.0f}  max={arr.max():.0f}")

        return bands

    # ──────────────────────────────────────────────────────────────────────────
    #  MANUAL IMAGE INPUT
    # ──────────────────────────────────────────────────────────────────────────    

    def load_bands(self) -> dict[str, np.ndarray]:
        """
        Load the required bands from TIFF files named

            <self.image_path>_1.tif
            <self.image_path>_2.tif
            ...

        Returns
        -------
        dict
            {"1": ndarray, "3": ndarray, ...}
        """
        prefix = Path(self.image_path)
        bands: dict[str, np.ndarray] = {}

        for band_key in sorted(self._needed_bands):
            filepath = prefix.parent / f"{prefix.name}_{band_key}.tif"

            if not filepath.exists():
                raise FileNotFoundError(f"Band {band_key} not found: {filepath}")

            arr = tifffile.imread(filepath).astype(float)

            # Remove singleton channel if present
            if arr.ndim == 3:
                arr = arr[:, :, 0]

            bands[band_key] = arr

            print(
                f"Band {band_key} ({self.BAND_LABELS.get(band_key, '?')}): "
                f"{arr.shape}  min={arr.min():.0f}  max={arr.max():.0f}"
            )

        return bands

    # ──────────────────────────────────────────────────────────────────────────
    #  CENTRE CROP
    # ──────────────────────────────────────────────────────────────────────────

    def crop_centre(
        self, bands: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], tuple[int, int, int, int]]:
        """Crop every band to the central (1 - 2*crop_fraction) region."""
        f      = self.crop_fraction
        sample = next(iter(bands.values()))
        h, w   = sample.shape

        r0 = int(h * f);      r1 = int(h * (1.0 - f))
        c0 = int(w * f);      c1 = int(w * (1.0 - f))

        cropped = {k: v[r0:r1, c0:c1] for k, v in bands.items()}
        print(f"  Crop [{r0}:{r1}, {c0}:{c1}]  {w}×{h} → {c1-c0}×{r1-r0} px")
        return cropped, (r0, r1, c0, c1)

    # ──────────────────────────────────────────────────────────────────────────
    #  BAND ALIGNMENT  (ECC — operates on already-cropped images)
    # ──────────────────────────────────────────────────────────────────────────

    def align_bands(
        self,
        bands: dict[str, np.ndarray],
        reference_band: str = "3",
    ) -> dict[str, np.ndarray]:
        """Align all bands to reference_band using ECC (Euclidean motion model)."""

        def to_uint8(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            return ((arr - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)

        if reference_band not in bands:
            reference_band = sorted(bands.keys())[0]
            print(f"  Reference band not available; using band {reference_band}.")

        ref_img  = to_uint8(bands[reference_band])
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_epsilon,
        )

        aligned = {reference_band: bands[reference_band].copy()}
        print("  Aligning bands (ECC on cropped region)…")

        for key, arr in bands.items():
            if key == reference_band:
                continue
            src  = to_uint8(arr)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp = cv2.findTransformECC(
                    ref_img, src, warp,
                    cv2.MOTION_EUCLIDEAN, criteria,
                    inputMask=None, gaussFiltSize=5,
                )
                h, w        = arr.shape
                aligned_arr = cv2.warpAffine(
                    arr.astype(np.float32), warp, (w, h),
                    flags      = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode = cv2.BORDER_REFLECT,
                )
                aligned[key] = aligned_arr.astype(float)
                tx    = warp[0, 2]; ty = warp[1, 2]
                angle = math.degrees(math.atan2(warp[1, 0], warp[0, 0]))
                print(f"    Band {key}: shift=({tx:+.2f}, {ty:+.2f}) px  "
                      f"rot={angle:+.3f}°")
            except cv2.error as e:
                print(f"    Band {key}: ECC failed — using unaligned. ({e})")
                aligned[key] = arr.copy()

        return aligned

    # ──────────────────────────────────────────────────────────────────────────
    #  INDEX COMPUTATION
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_ndvi(bands: dict) -> np.ndarray:
        """
        NDVI = (NIR - Red) / (NIR + Red)
        Bands used: 3 = Red (645 nm), 5 = NIR (900 nm)
        Range: −1 … +1. Healthy vegetation typically > 0.4.
        """
        nir = bands["5"]; red = bands["3"]
        return (nir - red) / (nir + red + 1e-6)

    @staticmethod
    def compute_pri(bands: dict) -> np.ndarray:
        """
        PRI = (Band1 - Band2) / (Band1 + Band2)
        Bands used: 1 = 528 nm (reference), 2 = 570 nm (xanthophyll-sensitive)
        Formula matches the user specification: (B1 - B2) / (B1 + B2)
        Range: ≈ −1 … +1. Healthy: > 0.05; stressed: < 0.05.
        """
        b1 = bands["1"]; b2 = bands["2"]
        return (b1 - b2) / (b1 + b2 + 1e-6)

    @staticmethod
    def build_rgb(bands: dict) -> np.ndarray:
        """
        Pseudo-colour RGB for display using bands 3 (R), 2 (G2), 1 (G).
        Applied percentile stretch (2nd–98th) per channel.
        """
        def stretch(arr):
            valid = arr[arr > 0]
            if valid.size == 0:
                return np.zeros_like(arr)
            p2, p98 = np.percentile(valid, (2, 98))
            return np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)
        return np.dstack([stretch(bands["3"]),   # R channel
                          stretch(bands["2"]),   # G channel
                          stretch(bands["1"])])  # B channel

    # ──────────────────────────────────────────────────────────────────────────
    #  SERIAL HELPER
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _nmea_checksum(payload: str) -> str:
        """Compute XOR checksum for NMEA-like payload string."""
        result = 0
        for c in payload:
            result ^= ord(c)
        return f"{result:02X}"

    # ──────────────────────────────────────────────────────────────────────────
    #  ATTITUDE-AWARE GEOTRANSFORM
    # ──────────────────────────────────────────────────────────────────────────
    
    def build_geo(
        self,
        lat_deg:   float,
        lon_deg:   float,
        alt_m:     float,
        full_h:    int,
        full_w:    int,
        crop_box:  tuple[int, int, int, int],
        attitude:  dict | None = None,
    ) -> dict:
        """
        Build the geotransform for the cropped image, incorporating heading.

        Heading (yaw) rotates the pixel axes so that column +1 points in the
        drone's forward direction rather than always East.  Roll and pitch
        are recorded in the output dict and flagged if excessive, but are not
        applied as a full perspective warp here (that would require a DEM).

        Parameters
        ----------
        attitude : dict with keys roll_deg, pitch_deg, yaw_deg  (all degrees)
                   If None, heading = 0° (North-up) is assumed.
        """
        r0, r1, c0, c1 = crop_box
        crop_h = r1 - r0
        crop_w = c1 - c0

        # Ground sampling distance (m/px) at nadir
        dist_drone_tree = alt_m - self.home_alt - self.tree_height
        gsd = (dist_drone_tree * self.sensor_width_mm) / (self.focal_length_mm * full_w)

        # Degrees per pixel at nadir (approximate flat-earth)
        gsd_lat = gsd / 111_320.0
        gsd_lon = gsd / (111_320.0 * math.cos(math.radians(lat_deg)) + 1e-9)

        # Heading: yaw measured CW from North → convert to CCW from East for math
        yaw_deg   = attitude["yaw_deg"]   if attitude else 0.0
        roll_deg  = attitude["roll_deg"]  if attitude else 0.0
        pitch_deg = attitude["pitch_deg"] if attitude else 0.0

        # Rotation matrix: maps image (col, row) offsets → (East, North) offsets
        # Image +col = drone right, image +row = drone forward (nadir, after yaw)
        # We treat yaw as the bearing of the +col (right) axis from East:
        #   heading 0° (North) → +col points East  → bearing = 90°
        #   heading 90° (East) → +col points South → bearing = 0°
        psi = math.radians(yaw_deg)
        # unit vectors (East, North) for image +col and +row axes
        col_north  =  -math.sin(psi)   # +col → right of drone
        col_east =  math.cos(psi)
        row_north  = -math.cos(psi)   # +row → down (away from nose)
        row_east =  -math.sin(psi)

        # Top-left corner of the crop in the full image (pixels from full-image centre)
        cr0_from_centre = r0 - full_h / 2
        cc0_from_centre = c0 - full_w / 2

        # Top-left corner in (East, North) metres from camera footprint centre
        tl_east  = (cc0_from_centre * col_east  + cr0_from_centre * row_east)  * gsd
        tl_north = (cc0_from_centre * col_north + cr0_from_centre * row_north) * gsd

        # Convert to geographic
        lat_top  = lat_deg + tl_north / 111_320.0
        lon_left = lon_deg + tl_east  / (111_320.0 * math.cos(math.radians(lat_deg)) + 1e-9)

        return {
            # Basic metrics
            "gsd_m":      gsd,
            "gsd_lat":    gsd_lat,
            "gsd_lon":    gsd_lon,
            "img_h":      crop_h,
            "img_w":      crop_w,
            # Top-left origin of the cropped image
            "lat_top":    lat_top,
            "lon_left":   lon_left,
            # Heading-aware pixel→geo basis vectors
            "col_east":   col_east,
            "col_north":  col_north,
            "row_east":   row_east,
            "row_north":  row_north,
            # Attitude metadata
            "yaw_deg":    yaw_deg,
            "roll_deg":   roll_deg,
            "pitch_deg":  pitch_deg,
        }

    def pixel_to_gps(self, row: float, col: float, geo: dict) -> tuple[float, float]:
        """
        Convert a (row, col) pixel in the **cropped** image to (lat, lon).

        Uses the heading-aware basis vectors stored in *geo*.
        """
        east  = (col * geo["col_east"]  + row * geo["row_east"])  * geo["gsd_m"]
        north = (col * geo["col_north"] + row * geo["row_north"]) * geo["gsd_m"]

        lat = geo["lat_top"] + north / 111_320.0
        lon = geo["lon_left"] + east / (
            111_320.0 * math.cos(math.radians(geo["lat_top"])) + 1e-9
        )
        return lat, lon

    # ──────────────────────────────────────────────────────────────────────────
    #  STRESS CLUSTER DETECTION  (per-index thresholds + radius health gate)
    # ──────────────────────────────────────────────────────────────────────────

    def find_stress_clusters(
        self,
        index_map:  np.ndarray,
        geo:        dict,
        index_name: str = "INDEX",
    ) -> tuple[pd.DataFrame, str]:
        """
        K-means clustering on stressed pixels.

        Returns
        -------
        df         : DataFrame of clusters
        health     : "UNHEALTHY" if any cluster radius > min_cluster_radius_m,
                     "HEALTHY"   otherwise
        """
        from sklearn.cluster import KMeans

        s_min, s_max = self.stress_thresholds.get(index_name, (0.10, 0.35))

        stressed = (
            (index_map >= s_min) &
            (index_map <  s_max) &
            (~np.isnan(index_map))
        )
        rows, cols = np.where(stressed)

        if len(rows) == 0:
            print(f"    No stressed pixels for {index_name} in [{s_min}, {s_max}).")
            return pd.DataFrame(), "HEALTHY"

        print(f"    {len(rows)} stressed pixels  "
              f"({index_name} ∈ [{s_min}, {s_max}))")

        n_k    = min(self.n_clusters, len(rows))
        coords = np.column_stack([cols, rows])
        km     = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)

        val_col  = f"mean_{index_name.lower()}"
        health   = "HEALTHY"
        candidates = []          # every cluster passing min_pixels

        for k in range(n_k):
            mask_k = labels == k
            if mask_k.sum() < self.min_pixels:
                continue

            cr = rows[mask_k]; cc = cols[mask_k]
            cv = index_map[cr, cc]

            cen_row = cr.mean(); cen_col = cc.mean()
            lat, lon = self.pixel_to_gps(cen_row, cen_col, geo)

            radius_px = max(cc.std(), cr.std())
            radius_m  = radius_px * geo["gsd_m"]

            mean_val = float(cv.mean())
            span = s_max - s_min + 1e-9
            if   mean_val < s_min + span * 0.25: severity = "CRITICAL"
            elif mean_val < s_min + span * 0.60: severity = "SEVERE"
            else:                                severity = "MODERATE"

            # ── radius gate: a cluster "satisfies the radius" when its radius
            #    reaches the minimum (radius_m >= min_cluster_radius_m). These are
            #    also the clusters that mark the tile UNHEALTHY.
            satisfies_radius = radius_m >= self.min_cluster_radius_m
            if satisfies_radius:
                health = "UNHEALTHY"

            # ── mean gate: only keep clusters whose CENTRE mean is still inside
            #    the stress window. (Always true while pixels are pre-filtered to
            #    [s_min, s_max); kept as an explicit safety guard.)
            mean_in_window = (s_min <= mean_val < s_max)

            candidates.append({
                "cluster_id":                k,
                "lat":                       round(lat, 7),
                "lon":                       round(lon, 7),
                "cen_row_px":                int(cen_row),
                "cen_col_px":                int(cen_col),
                val_col:                     round(mean_val, 4),
                f"min_{index_name.lower()}": round(float(cv.min()), 4),
                "pixel_count":               int(mask_k.sum()),
                "radius_px":                 round(radius_px, 1),
                "radius_m":                  round(radius_m, 2),
                "severity":                  severity,
                "exceeds_radius_gate":       satisfies_radius,
                "mean_in_window":            mean_in_window,
                "index":                     index_name,
            })

        if not candidates:
            print(f"    All clusters below min_pixels={self.min_pixels}.")
            return pd.DataFrame(), health

        # ── Select for output/telemetry ───────────────────────────────────────
        # Keep only clusters that satisfy the radius AND whose mean is in-window,
        # then take the n_output_clusters with the LOWEST mean index value
        # (most stressed).
        selected = [c for c in candidates
                    if c["exceeds_radius_gate"] and c["mean_in_window"]]
        selected.sort(key=lambda c: c[val_col])
        selected = selected[: self.n_output_clusters]

        if not selected:
            print(f"    {len(candidates)} cluster(s) formed, but none satisfy "
                  f"radius >= {self.min_cluster_radius_m} m with mean in "
                  f"[{s_min}, {s_max}).")
            return pd.DataFrame(), health

        df = (pd.DataFrame(selected)
                .sort_values(val_col)
                .reset_index(drop=True))

        print(f"    Formed {len(candidates)} cluster(s); outputting "
              f"{len(df)} (lowest-mean, radius-satisfying) of max "
              f"{self.n_output_clusters}.")
        print(f"\n    {'#':<4} {'Sev':<10} {'Lat':>11} {'Lon':>12} "
              f"{index_name:>7} {'Px':>6} {'r(m)':>7} {'⚠':>4}")
        print(f"    {'─'*64}")
        for _, row in df.iterrows():
            flag = "YES" if row.exceeds_radius_gate else ""
            print(f"    {int(row.cluster_id):<4} {row.severity:<10} "
                  f"{row.lat:>11.7f} {row.lon:>12.7f} "
                  f"{row[val_col]:>7.4f} {int(row.pixel_count):>6} "
                  f"{row.radius_m:>6.1f}m {flag:>4}")

        return df, health

    # ──────────────────────────────────────────────────────────────────────────
    #  PLOTTING
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_clusters(
        self, ax, clusters_df: pd.DataFrame, val_col: str
    ) -> None:
        if clusters_df is None or clusters_df.empty:
            return
        for _, row in clusters_df.iterrows():
            color  = self.SEVERITY_COLORS.get(row.severity, (1, 1, 0))
            r_px   = max(float(row.radius_px), 8)
            edge   = "white" if row.exceeds_radius_gate else color
            lw     = 3      if row.exceeds_radius_gate else 2
            circle = plt.Circle(
                (row.cen_col_px, row.cen_row_px), r_px,
                edgecolor=edge, facecolor=(*color, 0.2), linewidth=lw,
            )
            ax.add_patch(circle)
            ax.plot(row.cen_col_px, row.cen_row_px, '+',
                    color=color, markersize=10, markeredgewidth=2)
            gate_tag = " ⚠LARGE" if row.exceeds_radius_gate else ""
            # ax.annotate(
            #     f"{row.severity[0]}{gate_tag}  {row[val_col]:.2f}\n"
            #     f"{row.lat:.5f}\n{row.lon:.5f}",
            #     xy=(row.cen_col_px, row.cen_row_px),
            #     xytext=(6, 6), textcoords="offset points",
            #     fontsize=6.5, color="white",
            #     bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85),
            # )

    def plot_results(
        self,
        output:   str   = "micasense_result.png",
        lat_deg:  float = 0.0,
        lon_deg:  float = 0.0,
        alt_m:    float = 0.0,
        health_summary: dict | None = None,
    ) -> None:
        if self.last_rgb is None and not self.last_index_maps:
            raise RuntimeError("No data to plot — call run() first.")

        bands_crop = self.last_bands_crop
        index_maps = self.last_index_maps
        clusters   = self.last_clusters
        geo        = self.last_geo
        attitude   = self.last_attitude or {}

        has_rgb   = self.last_rgb is not None
        n_indices = len(index_maps)
        ncols     = (1 if has_rgb else 0) + n_indices
        fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 12))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        col_offset = 0

        if has_rgb:
            rgb = self.last_rgb
            axes[0, 0].imshow(rgb, origin="upper")
            axes[0, 0].set_title("Pseudo-RGB (G/G2/R, centre crop)", fontsize=12)
            axes[0, 0].axis("off")

            axes[1, 0].imshow(rgb, origin="upper")
            axes[1, 0].set_title("RGB + Stress Clusters", fontsize=12)
            axes[1, 0].axis("off")
            axes[1, 0].legend(
                handles=[
                    mpatches.Patch(color=(1, 0, 0),  label="Critical"),
                    mpatches.Patch(color=(1, .4, 0), label="Severe"),
                    mpatches.Patch(color=(1, .7, 0), label="Moderate"),
                    mpatches.Patch(edgecolor="white", facecolor="none",
                                   linewidth=2, label=f"> {self.min_cluster_radius_m} m (UNHEALTHY)"),
                ],
                fontsize=8, loc="lower left", framealpha=0.85,
            )
            col_offset = 1

        vmaxes = {"NDVI": 0.9, "PRI": 0.2}

        for i, (idx_name, idx_map) in enumerate(index_maps.items()):
            col     = i + col_offset
            val_col = f"mean_{idx_name.lower()}"
            df, _   = (clusters.get(idx_name, (pd.DataFrame(), "HEALTHY")))
            # unpack if tuple
            if isinstance(clusters.get(idx_name), tuple):
                df = clusters[idx_name][0]
            else:
                df = clusters.get(idx_name, pd.DataFrame())
            s_min, s_max = self.stress_thresholds.get(idx_name, (0.10, 0.35))
            vmax = vmaxes.get(idx_name, 0.5)

            im = axes[0, col].imshow(
                idx_map, cmap="RdYlGn", vmin=-0.1, vmax=vmax, origin="upper"
            )
            plt.colorbar(im, ax=axes[0, col], label=idx_name, shrink=0.65)
            axes[0, col].set_title(f"{idx_name} Map (centre crop)", fontsize=12)
            axes[0, col].axis("off")

            overlay = np.zeros((*idx_map.shape, 4))
            overlay[(idx_map >= s_min) & (idx_map < s_max)] = [1, 0.6, 0, 0.4]
            axes[0, col].imshow(overlay, origin="upper")

            im_cluster = axes[1, col].imshow(
                idx_map, cmap="RdYlGn", vmin=-0.1, vmax=vmax, origin="upper"
            )
            plt.colorbar(im_cluster, ax=axes[1, col], label=idx_name, shrink=0.65)
            health_label = health_summary.get(idx_name, "?") if health_summary else "?"
            axes[1, col].set_title(
                f"{idx_name} Clusters  [{s_min}–{s_max}]  →  {health_label}",
                fontsize=12,
                color="red" if health_label == "UNHEALTHY" else "green",
            )
            axes[1, col].axis("off")
            self._draw_clusters(axes[1, col], df, val_col)
            if has_rgb:
                self._draw_clusters(axes[1, 0], df, val_col)

        yaw   = attitude.get("yaw_deg",   0.0)
        roll  = attitude.get("roll_deg",  0.0)
        pitch = attitude.get("pitch_deg", 0.0)

        fig.text(
            0.01, 0.01,
            f"GSD ≈ {geo['gsd_m']*100:.1f} cm/px  |  Alt = {alt_m:.0f} m  |  "
            f"GPS ({lat_deg:.6f}°, {lon_deg:.6f}°)  |  "
            f"Heading={yaw:.1f}°  Roll={roll:+.1f}°  Pitch={pitch:+.1f}°  |  "
            f"Crop: [{self.crop_fraction*100:.0f}%–{(1-self.crop_fraction)*100:.0f}%]  |  "
            f"Bands: {sorted(self._needed_bands)}  |  "
            f"Radius gate: >{self.min_cluster_radius_m} m → UNHEALTHY",
            fontsize=7.5, color="grey",
        )

        # Overall health banner
        if health_summary:
            overall = "UNHEALTHY" if "UNHEALTHY" in health_summary.values() else "HEALTHY"
            fig.text(
                0.5, 0.005, f"Overall assessment: {overall}",
                ha="center", fontsize=13, fontweight="bold",
                color="red" if overall == "UNHEALTHY" else "green",
            )

        plt.suptitle("MicaSense RedEdge — Live Capture Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"  Saved {output}")
        # plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        lat_deg:   float = -1000.0,
        lon_deg:   float = -1000.0,
        alt_m:     float = -1000.0,
        save_csv:  str   = "stress_clusters.csv",
        save_plot: str   = "micasense_result.png",
    ) -> dict:
        """
        Execute the full pipeline for one capture.

        Pipeline
        --------
        1.  Obtain GPS fix
        2.  Obtain attitude (roll, pitch, yaw from DLS IMU)
        3.  Trigger capture  →  download only required bands
        4.  Crop to centre   →  before alignment (reduces ECC cost ~75 %)
        5.  Align bands      →  ECC on cropped images
        6.  Build RGB preview (pseudo-colour if bands 1–3 downloaded)
        7.  Compute indices  →  NDVI, PRI, or both
        8.  Build attitude-aware geotransform
        9.  Detect stress clusters (per-index thresholds)
        10. Apply radius gate  →  HEALTHY / UNHEALTHY per index
        11. Save CSV + plot
        """
        sep = "─" * 60
        t0  = time.perf_counter()

        # 1. GPS
        print(f"\n{sep}\nStep 1 — Obtain GPS")
        lat_r, lon_r, alt_r = self.obtain_gps()
        if lat_deg == -1000.0 or lon_deg == -1000.0:
            lat_deg, lon_deg = lat_r, lon_r
        if alt_m == -1000.0:
            alt_m = alt_r

        # 2. Attitude
        print(f"\n{sep}\nStep 2 — Obtain attitude (roll / pitch / yaw)")
        attitude = self.obtain_attitude()
        self.last_attitude = attitude

        # 3. Capture
        print(f"\n{sep}\nStep 3 — Capture & selective band download "
              f"(bands {sorted(self._needed_bands)})")
        t_cap     = time.perf_counter()
        bands_raw = self.trigger_capture()
        print(f"  ↳ download took {time.perf_counter()-t_cap:.2f} s")

        sample      = next(iter(bands_raw.values()))
        full_h, full_w = sample.shape

        # 4. Crop
        print(f"\n{sep}\nStep 4 — Centre crop "
              f"({int(self.crop_fraction*100)}%–{int((1-self.crop_fraction)*100)}%)")
        bands_crop, crop_box = self.crop_centre(bands_raw)
        del bands_raw

        # 5. Align
        print(f"\n{sep}\nStep 5 — Band alignment (ECC)")
        t_ecc = time.perf_counter()
        # Use band "3" (Red 645 nm) as reference; fall back if not downloaded
        bands_aligned = self.align_bands(bands_crop, reference_band="3")
        print(f"  ↳ ECC took {time.perf_counter()-t_ecc:.2f} s")

        # 6. RGB
        has_rgb = {"1", "2", "3"}.issubset(bands_aligned.keys())
        rgb     = self.build_rgb(bands_aligned) if has_rgb else None
        if not has_rgb:
            print("  RGB preview skipped (need bands 1, 2, 3).")

        # 7. Indices
        print(f"\n{sep}\nStep 6 — Index computation ({self.index_mode})")
        index_maps: dict[str, np.ndarray] = {}

        if self.index_mode in ("ndvi", "both"):
            if {"3", "5"}.issubset(bands_aligned):
                ndvi = self.compute_ndvi(bands_aligned)
                index_maps["NDVI"] = ndvi
                s = self.stress_thresholds["NDVI"]
                print(f"  NDVI range: {np.nanmin(ndvi):.3f} → {np.nanmax(ndvi):.3f}  "
                      f"stress=[{s[0]}, {s[1]})")
            else:
                print("  NDVI skipped — bands 3 or 5 not downloaded.")

        if self.index_mode in ("pri", "both"):
            if {"1", "2"}.issubset(bands_aligned):
                pri = self.compute_pri(bands_aligned)
                index_maps["PRI"] = pri
                s = self.stress_thresholds["PRI"]
                print(f"  PRI  range: {np.nanmin(pri):.3f} → {np.nanmax(pri):.3f}  "
                      f"stress=[{s[0]}, {s[1]})")
            else:
                print("  PRI skipped — bands 1 or 2 not downloaded.")

        if not index_maps:
            raise ValueError(
                f"No indices could be computed for index_mode='{self.index_mode}'. "
                "Check that required bands were downloaded."
            )

        # 8. Attitude-aware geotransform
        print(f"\n{sep}\nStep 7 — Attitude-aware geotransform")
        geo = self.build_geo(
            lat_deg, lon_deg, alt_m,
            full_h, full_w, crop_box,
            attitude=attitude,
        )
        print(f"  Full image  : {full_w} × {full_h} px")
        print(f"  Crop region : {geo['img_w']} × {geo['img_h']} px")
        print(f"  GSD         ≈ {geo['gsd_m']*100:.1f} cm/px")
        print(f"  Heading     : {geo['yaw_deg']:.1f}°  "
              f"Roll: {geo['roll_deg']:+.2f}°  Pitch: {geo['pitch_deg']:+.2f}°")

        # 9 & 10. Clusters + health gate
        print(f"\n{sep}\nStep 8 — Stress clusters + radius health gate "
              f"(gate > {self.min_cluster_radius_m} m)")
        clusters: dict[str, tuple[pd.DataFrame, str]] = {}
        health_summary: dict[str, str] = {}
        all_rows = []

        for idx_name, idx_map in index_maps.items():
            s = self.stress_thresholds[idx_name]
            print(f"\n  [{idx_name}]  threshold=[{s[0]}, {s[1]})")
            df, health = self.find_stress_clusters(idx_map, geo, index_name=idx_name)
            clusters[idx_name] = (df, health)
            health_summary[idx_name] = health
            print(f"  [{idx_name}] health assessment → {health}")
            if not df.empty:
                all_rows.append(df)

        # Overall
        overall = "UNHEALTHY" if "UNHEALTHY" in health_summary.values() else "HEALTHY"
        print(f"\n{'═'*60}")
        print(f"  OVERALL HEALTH ASSESSMENT:  {overall}")
        print(f"{'═'*60}")
        for idx_name, h in health_summary.items():
            print(f"    {idx_name:<6}: {h}")

        # 11. Save CSV
        if save_csv and all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_csv(save_csv, index=False)
            print(f"\n  Saved {save_csv}")

        # Persist state
        self.last_bands_crop = bands_aligned
        self.last_rgb        = rgb
        self.last_index_maps = index_maps
        self.last_clusters   = clusters
        self.last_geo        = geo

        elapsed = time.perf_counter() - t0
        print(f"\n{sep}\nTotal pipeline time: {elapsed:.2f} s")

        # 12. Plot
        if save_plot:
            print(f"\n{sep}\nStep 9 — Plot")
            self.plot_results(
                output=save_plot,
                lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m,
                health_summary=health_summary,
            )
        
        # 13. Transmit image-centre GPS and all cluster centres in one message.
        print(f"\n{sep}\nStep 10 — Transmit image-centre GPS and cluster locations "
            f"({self.serial_port} @ {self.serial_baud} baud)")

        # ── Build the payload ──────────────────────────────────────────────────────
        # Format:
        #   <img_lat>,<img_lon>,<img_alt>,<c1_lat>,<c1_lon>,<c2_lat>,<c2_lon>,...
        # First three fields are the image-centre GPS fix (lat, lon, alt).
        # Each subsequent pair is the 2-D geographic centre of one stress cluster.

        cluster_pairs = [
            (row["lat"], row["lon"])
            for df_cluster, _health in clusters.values()
            if not df_cluster.empty
            for _, row in df_cluster.iterrows()
        ]

        # Image-centre coordinates come from the geotransform built earlier.
        # geo["lat_top"] and geo["lon_left"] define the top-left corner of the
        # crop; the centre is half the crop dimensions away from that corner.
        img_lat = lat_deg
        img_lon = lon_deg
        img_alt = alt_m

        # Assemble payload string
        header  = f"{img_lat:.6f},{img_lon:.6f},{img_alt:.2f}"
        cluster_str = ",".join(f"{lat:.6f},{lon:.6f}" for lat, lon in cluster_pairs)
        payload = f"{header},{cluster_str}" if cluster_pairs else header

        msg = f"{payload}*{self._nmea_checksum(payload)}\n"

        # ── Transmit ───────────────────────────────────────────────────────────────
        try:
            with serial.Serial(self.serial_port, self.serial_baud, timeout=1) as ser:
                if self.serial_settle_s:
                    time.sleep(self.serial_settle_s)
                ser.write(msg.encode("utf-8"))
                print(f"  Sent : {msg.strip()}")
                print(f"  Image centre : ({img_lat:.6f}, {img_lon:.6f}, {img_alt:.2f} m)")
                print(f"  Clusters     : {len(cluster_pairs)}")
        except serial.SerialException as e:
            print(f"  Serial error: {e}")


        return {
            "bands_crop":     bands_aligned,
            "rgb":            rgb,
            "index_maps":     index_maps,
            "clusters":       clusters,
            "health_summary": health_summary,
            "overall_health": overall,
            "geo":            geo,
            "attitude":       attitude,
        }
    

    def run_standalone(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        save_csv: str = "stress_clusters.csv",
        save_plot: str = "micasense_result.png",
    ) -> dict:
        """
        Execute the processing pipeline using TIFF files stored on disk.

        Assumes self.image_path points to files named

            <image_path>_1.tif
            <image_path>_2.tif
            ...

        GPS and altitude are supplied by the caller.
        """
        sep = "─" * 60
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Load bands
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 1 — Load TIFF bands")
        t_load = time.perf_counter()
        bands_raw = self.load_bands()
        print(f"  ↳ loading took {time.perf_counter() - t_load:.2f} s")

        sample = next(iter(bands_raw.values()))
        full_h, full_w = sample.shape

        # ------------------------------------------------------------------
        # 2. Crop
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 2 — Centre crop")
        bands_crop, crop_box = self.crop_centre(bands_raw)
        del bands_raw

        # ------------------------------------------------------------------
        # 3. Align
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 3 — Band alignment (ECC)")
        t_ecc = time.perf_counter()
        bands_aligned = self.align_bands(bands_crop, reference_band="3")
        print(f"  ↳ ECC took {time.perf_counter() - t_ecc:.2f} s")

        # ------------------------------------------------------------------
        # 4. RGB preview
        # ------------------------------------------------------------------
        has_rgb = {"1", "2", "3"}.issubset(bands_aligned)
        rgb = self.build_rgb(bands_aligned) if has_rgb else None

        if not has_rgb:
            print("  RGB preview skipped (need bands 1,2,3).")

        # ------------------------------------------------------------------
        # 5. Compute indices
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 4 — Index computation ({self.index_mode})")

        index_maps = {}

        if self.index_mode in ("ndvi", "both"):
            if {"3", "5"}.issubset(bands_aligned):
                ndvi = self.compute_ndvi(bands_aligned)
                index_maps["NDVI"] = ndvi

        if self.index_mode in ("pri", "both"):
            if {"1", "2"}.issubset(bands_aligned):
                pri = self.compute_pri(bands_aligned)
                index_maps["PRI"] = pri

        if not index_maps:
            raise ValueError("No vegetation indices could be computed.")

        # ------------------------------------------------------------------
        # 6. Geotransform
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 5 — Geotransform")

        # No attitude available for standalone mode.
        attitude = {
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        }

        geo = self.build_geo(
            lat_deg,
            lon_deg,
            alt_m,
            full_h,
            full_w,
            crop_box,
            attitude=attitude,
        )

        # ------------------------------------------------------------------
        # 7. Cluster detection
        # ------------------------------------------------------------------
        print(f"\n{sep}\nStep 6 — Stress clusters")

        clusters = {}
        health_summary = {}
        all_rows = []

        for idx_name, idx_map in index_maps.items():
            df, health = self.find_stress_clusters(
                idx_map,
                geo,
                index_name=idx_name,
            )

            clusters[idx_name] = (df, health)
            health_summary[idx_name] = health

            if not df.empty:
                all_rows.append(df)

        overall = (
            "UNHEALTHY"
            if "UNHEALTHY" in health_summary.values()
            else "HEALTHY"
        )

        # ------------------------------------------------------------------
        # 8. Save CSV
        # ------------------------------------------------------------------
        if save_csv and all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(save_csv, index=False)
            print(f"Saved {save_csv}")

        # ------------------------------------------------------------------
        # Persist state
        # ------------------------------------------------------------------
        self.last_bands_crop = bands_aligned
        self.last_rgb = rgb
        self.last_index_maps = index_maps
        self.last_clusters = clusters
        self.last_geo = geo
        self.last_attitude = attitude

        print(f"\n{sep}")
        print(f"Total pipeline time: {time.perf_counter() - t0:.2f} s")

        # ------------------------------------------------------------------
        # 9. Plot
        # ------------------------------------------------------------------
        if save_plot:
            self.plot_results(
                output=save_plot,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_m=alt_m,
                health_summary=health_summary,
            )

        return {
            "bands_crop": bands_aligned,
            "rgb": rgb,
            "index_maps": index_maps,
            "clusters": clusters,
            "health_summary": health_summary,
            "overall_health": overall,
            "geo": geo,
            "attitude": attitude,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE USAGE GIVEN IMAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    analyzer = MicaSenseAnalyzer(
        cache_format          = "tif",      # "jpeg" (fast 8-bit) or "tif" (16-bit)
        index_mode            = "both",      # "ndvi" | "pri" | "both"
        want_rgb              = True,

        # NDVI stress window  (healthy vegetation typically > 0.4)
        ndvi_stress_min       = 0.10,  # -1.0, # 0.10, 
        ndvi_stress_max       = 0.40,  # 0.0,  # 0.40,

        # PRI stress window   (healthy > 0.05; stressed < 0.05)
        pri_stress_min        = -0.20, # -1.0, # -0.20,
        pri_stress_max        = 0.05,  # 0.0,  # 0.05,

        min_cluster_radius_m  = 0.3,

        n_clusters            = 4,
        n_output_clusters     = 2,
        min_pixels            = 30,
        crop_fraction         = 0.0, 
        image_path            = "test_images_turkey/EXP_TURKEY"
    )

    result = analyzer.run_standalone(
        # Override GPS/alt:
        lat_deg  = 42.000000,
        lon_deg  = -80.000000,
        alt_m    = 30.0,
        save_csv  = "stress_clusters.csv",
        save_plot = "micasense_result.png",
    )

    print("\nOverall health:", result["overall_health"])

# # ══════════════════════════════════════════════════════════════════════════════
# #  LOOP USAGE
# # ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":

#     # ── Operating-altitude gate ──────────────────────────────────────────────
#     # The pipeline runs only while GPS altitude is at/above this value (metres).
#     # Below it, the script idles and re-polls GPS every POLL_INTERVAL_S seconds.
#     gps_instant = MicaSenseAnalyzer(camera_ip = "http://192.168.1.83")
#     home_lat, home_lon, home_alt = gps_instant.obtain_gps()
#     HOME_ALT_M              = home_alt       # ← set your home altitude here (the "yyy m")
#     MIN_OPERATING_ALT_AGL_M = 25.0      # ← set your threshold here (the "xxx m")
#     POLL_INTERVAL_S         = 5.0       # how often to re-check GPS while grounded

#     analyzer = MicaSenseAnalyzer(
#         camera_ip             = "http://192.168.1.83",
#         cache_format          = "tif",      # "jpeg" (fast 8-bit) or "tif" (16-bit)
#         index_mode            = "ndvi",      # "ndvi" | "pri" | "both"
#         want_rgb              = False,
#         ndvi_stress_min       = 0.10,
#         ndvi_stress_max       = 0.60,
#         pri_stress_min        = -0.20,
#         pri_stress_max        = 0.05,
#         min_cluster_radius_m  = 4.0,
#         n_clusters            = 10,
#         n_output_clusters     = 2,
#         min_pixels            = 30,
#         crop_fraction         = 0.25,
#         max_tilt_deg          = 15.0,
#         tree_height           = 0.0,
#         home_alt              = HOME_ALT_M, 
#         serial_port           = "/dev/ttyUSB0",
#         serial_baud           = 57600,
#         serial_settle_s       = 0.02         # seconds to wait before writing
#     )

#     print(f"\nWaiting for altitude ≥ {MIN_OPERATING_ALT_AGL_M} m to begin captures. "
#           f"Ctrl+C to stop.")

#     capture_idx = 0
#     try:
#         while True:
#             # Cheap GPS poll to gate BEFORE the expensive capture/align/analysis.
#             try:
#                 lat, lon, alt = analyzer.obtain_gps()
#             except Exception as e:
#                 print(f"  GPS poll failed: {e} — retry in {POLL_INTERVAL_S}s.")
#                 time.sleep(POLL_INTERVAL_S)
#                 continue

#             if alt < HOME_ALT_M + MIN_OPERATING_ALT_AGL_M:
#                 print(f"  Altitude {alt:.1f} m < {HOME_ALT_M + MIN_OPERATING_ALT_AGL_M} m — idle. "
#                       f"Re-checking in {POLL_INTERVAL_S}s.")
#                 time.sleep(POLL_INTERVAL_S)
#                 continue

#             # Above threshold → run one full capture/analysis cycle.
#             capture_idx += 1
#             stamp = time.strftime("%Y%m%d_%H%M%S")
#             print(f"\n{'#'*60}\n# Capture {capture_idx}  "
#                   f"(alt {alt:.1f} m ≥ {HOME_ALT_M + MIN_OPERATING_ALT_AGL_M} m)  {stamp}\n{'#'*60}")

#             try:
#                 result = analyzer.run(
#                     save_csv  = f"results/stress_clusters_{capture_idx}_{stamp}.csv",
#                     save_plot = f"results/micasense_result_{capture_idx}_{stamp}.png",
#                 )
#                 print(f"\nCapture {capture_idx} overall health: "
#                       f"{result['overall_health']}")
#             except Exception as e:
#                 print(f"  Capture {capture_idx} failed: {e} — continuing loop.")

#     except KeyboardInterrupt:
#         print(f"\nStopped after {capture_idx} capture(s).")
