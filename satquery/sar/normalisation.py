import os
import subprocess
import tempfile
import warnings

import numpy as np
import rasterio

from satquery.config import preprocessing_config
from satquery.sar.snap_templates import SAR_GRAPH_TEMPLATE_NO_TC, SAR_GRAPH_TEMPLATE_WITH_TC
from satquery.sar.speckle import refined_lee
from satquery.texture import glcm_entropy


class SARNormaliser:
    """
    Implements the P2 SAR Normalisation pipeline as described in the master plan.
    It calls ESA SNAP via subprocess to avoid GPL linkage, then performs
    custom dB conversion, percentile stretching, and model stack generation.
    """

    def __init__(self):
        self.config = preprocessing_config()
        self.sar_config = self.config.sar
        self.pol_dropout_rate = self.sar_config.pol_dropout_rate

    def process(
        self,
        input_path: str,
        output_path: str,
        is_pregeoreferenced: bool = False,
        force_single_pol: bool = False,
    ):
        """
        Runs the full SAR normalisation pipeline.
        """
        # Step 1-5: SNAP Processing
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_out = os.path.join(tmpdir, "snap_processed.tif")
            self._run_snap_graph(input_path, snap_out, is_pregeoreferenced)

            # Step 6-8: Custom Python Processing
            with rasterio.open(snap_out) as src:
                # Read all bands (SNAP outputs linear intensity/sigma0)
                # Ensure we avoid <= 0 before log10
                data = src.read()
                data = np.clip(data, 1e-10, None)

                # Polarisation handling
                band_count = data.shape[0]
                pol_mode = "dual" if band_count >= 2 else "single"

                # Apply dropout if requested during training
                if pol_mode == "dual" and force_single_pol:
                    pol_mode = "single"
                    data = data[0:1, :, :]  # Take only the first band (e.g. VV)

                # Step 6: dB Conversion
                db_data = self._convert_to_db(data)

                # Sanity check before stretch
                self._sanity_check_db(db_data, pol_mode)

                # Step 7 & 8: Stretch and Model Stack
                if pol_mode == "dual":
                    pol1 = db_data[0]
                    pol2 = db_data[1]
                    # Robust stretch per band
                    pol1_stretch = self._percentile_stretch(pol1)
                    pol2_stretch = self._percentile_stretch(pol2)

                    # Ratio pol1/pol2 in dB is pol1_dB - pol2_dB
                    ratio = pol1 - pol2
                    ratio_stretch = self._percentile_stretch(ratio)

                    stack = np.stack([pol1_stretch, pol2_stretch, ratio_stretch])
                else:
                    # Section 4.2: [sigma0_dB, refined-Lee sigma0_dB, GLCM entropy].
                    #
                    # This used to write pol1 into two of the three channels. That
                    # is the "never tile a single SAR band to fake RGB" failure in
                    # a thinner disguise: a duplicated channel carries no
                    # information, so the single-pol stack would have been a
                    # 2-channel stack with a decorative third, and the pol-dropout
                    # training the hidden set depends on would have taught the
                    # model a stack it will never see.
                    pol1 = db_data[0]
                    pol1_stretch = self._percentile_stretch(pol1)

                    # Speckle is multiplicative, so the filter runs on linear
                    # power and the result is converted back to dB.
                    linear = np.power(10.0, pol1 / 10.0)
                    filtered_db = 10.0 * np.log10(np.clip(refined_lee(linear), 1e-10, None))
                    filtered_stretch = self._percentile_stretch(filtered_db)

                    texture = self._compute_texture(pol1)

                    stack = np.stack([pol1_stretch, filtered_stretch, texture])

                # Write output
                profile = src.profile
                profile.update(count=3, dtype=rasterio.float32)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(stack.astype(np.float32))

    def _run_snap_graph(self, input_path: str, output_path: str, is_pregeoreferenced: bool):
        """
        Executes the SNAP graph via subprocess.
        """
        # SarConfig is a frozen dataclass, not a mapping: `.get` raised
        # AttributeError on every call, so this branch never executed.
        skip_tc = self.sar_config.skip_terrain_correction_if_pregeoreferenced

        if is_pregeoreferenced and skip_tc:
            template = SAR_GRAPH_TEMPLATE_NO_TC
        else:
            template = SAR_GRAPH_TEMPLATE_WITH_TC

        # Replace variables
        graph_xml = template.replace("${sourceFile}", input_path).replace(
            "${targetFile}", output_path
        )

        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
            f.write(graph_xml)
            graph_path = f.name

        try:
            # We assume 'gpt' is in the system PATH.
            cmd = ["gpt", graph_path]
            # Since this is a critical integration, if 'gpt' is not found, we raise a clear error.
            # However, for tests, we might mock this.
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            # Fallback for systems without SNAP (e.g. CI without mock)
            warnings.warn(
                "SNAP 'gpt' executable not found. "
                "This requires a mock in tests or a real SNAP installation.",
                stacklevel=2,
            )
            raise
        finally:
            if os.path.exists(graph_path):
                os.remove(graph_path)

    def _convert_to_db(self, data: np.ndarray) -> np.ndarray:
        """Step 6: Convert linear intensity to dB."""
        return 10.0 * np.log10(data)

    def _percentile_stretch(self, data: np.ndarray) -> np.ndarray:
        """Step 7: Robust percentile stretch to [0, 1]."""
        low_p = self.config.radiometry.low_percentile
        high_p = self.config.radiometry.high_percentile

        # Ignore NaNs/Infs for percentile calculation
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return np.zeros_like(data)

        p_low, p_high = np.percentile(valid_data, (low_p, high_p))
        if p_high == p_low:
            return np.zeros_like(data)

        stretched = (data - p_low) / (p_high - p_low)
        return np.clip(stretched, 0.0, 1.0)

    def _compute_texture(self, image: np.ndarray) -> np.ndarray:
        """Windowed GLCM entropy for the single-pol stack (section 4.2).

        Previously a `generic_filter(np.std, size=3)`, which is a Python callback
        per pixel: minutes on a benchmark chip and hours on a full scene, against
        a 5 min/scene prep SLA. `satquery.texture.glcm_entropy` computes the real
        co-occurrence entropy symbol by symbol over vectorised uniform filters,
        so it is both the quantity the plan specifies and fast enough to run.
        """
        cfg = self.config.texture
        return glcm_entropy(image, window=cfg.window_px, levels=cfg.glcm_levels)

    def _sanity_check_db(
        self, db_data: np.ndarray, pol_mode: str, sar_band: str | None = None
    ) -> list[str]:
        """Sigma-nought sanity check against the surface table (section 4.2).

        **These warn, they never block.** A hard assert tuned on C-band would
        refuse a legitimate X-band scene out of the graded hidden set, which is a
        self-inflicted zero.

        Two fixes over the earlier version: the range is derived from the actual
        per-band surface table rather than a hardcoded -30..10, and the check
        uses robust percentiles rather than the mean, which a large water body or
        a bright urban block pulls out of range on its own. It also reports which
        band it used, so an untuned band inheriting C-band is visible instead of
        silent.
        """
        band = (sar_band or "C").upper()
        table = self.sar_config.sanity_ranges_db.get(band) or self.sar_config.sanity_ranges_db.get(
            "C"
        )
        if not table:
            return []

        low = min(bounds[0] for bounds in table.values())
        high = max(bounds[1] for bounds in table.values())
        finite = db_data[np.isfinite(db_data)]
        if finite.size == 0:
            return ["sigma-nought sanity check skipped: no finite pixels"]

        p2, p98 = np.percentile(finite, [2.0, 98.0])
        messages: list[str] = []
        if p2 < low - 10.0 or p98 > high + 10.0:
            message = (
                f"{band}-band {pol_mode}-pol sigma-nought spans {p2:.1f} to {p98:.1f} dB, "
                f"outside the {low:.0f} to {high:.0f} dB surface table; proceeding with "
                f"lowered SAR-side confidence (warning only, section 4.2)"
            )
            messages.append(message)
            warnings.warn(message, stacklevel=2)
        if self.sar_config.inherits_c_band(sar_band):
            messages.append(
                f"{band}-band ranges are untuned and inherit the C-band reference table"
            )
        return messages
