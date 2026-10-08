"""
Statistical inference, hypothesis testing, effect sizes, and Spatial Effective Sample Size (ESS).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import geopandas as gpd

from geoflow.spatial.autocorrelation import morans_i


class StatTestResult:
    """Encapsulates statistical test result, effect size, and confidence intervals."""

    def __init__(
        self,
        test_name: str,
        statistic: float,
        p_value: float,
        effect_size: float,
        effect_type: str,
        ci_95: Tuple[float, float],
        sample_sizes: Dict[str, int],
        spatial_warning: Optional[str] = None,
    ):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.effect_size = effect_size
        self.effect_type = effect_type
        self.ci_95 = ci_95
        self.sample_sizes = sample_sizes
        self.spatial_warning = spatial_warning

    def summary(self) -> str:
        sig = "Statistically Significant (p < 0.05)" if self.p_value < 0.05 else "Not Significant (p >= 0.05)"
        msg = (
            f"=== {self.test_name.upper()} ===\n"
            f"Statistic         : {self.statistic:.4f}\n"
            f"p-value           : {self.p_value:.6e} ({sig})\n"
            f"Effect Size ({self.effect_type}) : {self.effect_size:.4f}\n"
            f"95% CI            : [{self.ci_95[0]:.4f}, {self.ci_95[1]:.4f}]\n"
            f"Sample Sizes      : {self.sample_sizes}\n"
        )
        if self.spatial_warning:
            msg += f"\nWARNING (Spatial Autocorrelation Alert):\n{self.spatial_warning}\n"
        return msg

    def __repr__(self) -> str:
        return f"<StatTestResult {self.test_name} p={self.p_value:.4e} {self.effect_type}={self.effect_size:.3f}>"


def effective_sample_size(
    data: Any,
    value: str = "rh98",
    geometry: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute Effective Sample Size (ESS) adjusting for spatial autocorrelation.
    Prevents false certainty and pseudoreplication in remote sensing studies.
    N_eff = N * (1 - rho) / (1 + rho)
    """
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
    elif hasattr(data, "gdf"):
        gdf = data.gdf
    else:
        raise TypeError("effective_sample_size requires GeoDataFrame or FeatureCollection")

    n = len(gdf)
    moran = morans_i(gdf, value=value)
    rho = max(0.0, min(0.99, moran.I))  # Positive autocorrelation dampens effective N

    n_eff = n * (1.0 - rho) / (1.0 + rho) if (1.0 + rho) > 0 else float(n)
    design_effect = n / max(1.0, n_eff)

    pseudoreplication = (design_effect > 2.0 and moran.p_value < 0.05)

    return {
        "nominal_n": n,
        "effective_n": int(np.round(n_eff)),
        "morans_i": moran.I,
        "p_value": moran.p_value,
        "design_effect": design_effect,
        "pseudoreplication_risk": pseudoreplication,
    }


def test(
    data: Any,
    value: str,
    group: str,
    method: Optional[str] = None,
) -> StatTestResult:
    """
    Perform two-sample or multi-sample hypothesis tests with effect sizes and spatial warnings.
    """
    gdf = data.gdf if hasattr(data, "gdf") else data
    groups = gdf[group].unique()

    if len(groups) == 2:
        g1_data = gdf[gdf[group] == groups[0]][value].dropna().to_numpy()
        g2_data = gdf[gdf[group] == groups[1]][value].dropna().to_numpy()

        # Check spatial autocorrelation
        ess1 = effective_sample_size(gdf, value=value)
        spatial_warn = None
        if ess1["pseudoreplication_risk"]:
            spatial_warn = (
                f"Significant spatial autocorrelation detected (Moran's I={ess1['morans_i']:.3f}). "
                f"Nominal N={ess1['nominal_n']} but Effective N={ess1['effective_n']}. "
                "The degrees of freedom are inflated by spatial pseudoreplication. "
                "Consider spatially blocked permutation tests or spatial regression."
            )

        if method in ("mann_whitney", "mwu"):
            stat, p = scipy.stats.mannwhitneyu(g1_data, g2_data)
            # Rank-biserial correlation
            r = 1.0 - (2.0 * stat) / (len(g1_data) * len(g2_data))
            return StatTestResult("Mann-Whitney U", stat, p, r, "rank_biserial", (0.0, 0.0), {str(groups[0]): len(g1_data), str(groups[1]): len(g2_data)}, spatial_warn)
        else:
            # Welch's t-test (default)
            stat, p = scipy.stats.ttest_ind(g1_data, g2_data, equal_var=False)
            # Cohen's d
            s_pooled = np.sqrt(((len(g1_data) - 1) * np.var(g1_data, ddof=1) + (len(g2_data) - 1) * np.var(g2_data, ddof=1)) / (len(g1_data) + len(g2_data) - 2))
            d = (np.mean(g1_data) - np.mean(g2_data)) / s_pooled if s_pooled > 0 else 0.0

            diff = np.mean(g1_data) - np.mean(g2_data)
            se_diff = np.sqrt(np.var(g1_data, ddof=1)/len(g1_data) + np.var(g2_data, ddof=1)/len(g2_data))
            ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

            return StatTestResult("Welch t-test", stat, p, d, "cohens_d", ci, {str(groups[0]): len(g1_data), str(groups[1]): len(g2_data)}, spatial_warn)
    else:
        # One-way ANOVA
        group_arrays = [gdf[gdf[group] == g][value].dropna().to_numpy() for g in groups]
        stat, p = scipy.stats.f_oneway(*group_arrays)
        # Eta-squared
        all_vals = np.concatenate(group_arrays)
        ss_total = np.sum((all_vals - np.mean(all_vals))**2)
        ss_between = sum(len(arr) * (np.mean(arr) - np.mean(all_vals))**2 for arr in group_arrays)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0

        return StatTestResult("One-way ANOVA", stat, p, eta2, "eta_squared", (0.0, 0.0), {str(g): len(arr) for g, arr in zip(groups, group_arrays)})


class Power:
    """Power analysis and sample size estimation helper."""

    @staticmethod
    def sample_size(effect_size: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
        """Estimate required sample size per group for two-sample t-test."""
        # Standard normal approximation: n = 2 * ((z_alpha + z_power) / d)^2
        z_alpha = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        z_power = scipy.stats.norm.ppf(power)
        n = 2.0 * ((z_alpha + z_power) / effect_size)**2
        return int(np.ceil(n))


power = Power()
