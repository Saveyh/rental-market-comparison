from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS = {
    "id",
    "name",
    "host_id",
    "host_name",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "number_of_reviews_ltm",
    "license",
}

OUTPUT_COLUMNS = [
    "source_listing_id",
    "host_id",
    "city",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "price",
    "price_imputed",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "last_review_year",
    "reviews_per_month",
    "reviews_per_month_imputed",
    "calculated_host_listings_count",
    "availability_365",
    "number_of_reviews_ltm",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = {
    "Zurich": PROJECT_ROOT / "data" / "raw" / "zurich_listings.csv",
    "Milan": PROJECT_ROOT / "data" / "raw" / "milan_listings.csv",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "zurich_milan_listings_clean.csv"


def load_city_dataset(path: Path, city: str) -> pd.DataFrame:
    """Load a city extract and validate the expected schema."""
    dataset = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    dataset = dataset.copy()
    dataset["city"] = city
    return dataset


def build_combined_dataset(source_files: Mapping[str, Path] | None = None) -> pd.DataFrame:
    """Build a cleaned, analysis-ready dataset from the raw city extracts."""
    city_sources = source_files or DEFAULT_SOURCES
    frames = [load_city_dataset(Path(path), city) for city, path in city_sources.items()]
    combined = pd.concat(frames, ignore_index=True)

    combined = combined.rename(columns={"id": "source_listing_id"})
    combined = combined.drop(columns=["name", "host_name", "license"])

    combined["last_review"] = pd.to_datetime(combined["last_review"], errors="coerce")
    combined["price_imputed"] = combined["price"].isna()
    combined["reviews_per_month_imputed"] = combined["reviews_per_month"].isna()

    combined["price"] = _impute_price(combined)
    combined["reviews_per_month"] = combined["reviews_per_month"].fillna(0.0)

    combined["minimum_nights"] = combined["minimum_nights"].clip(upper=365)
    combined["price"] = _clip_by_city_quantile(combined, "price", 0.99).round(2)
    combined["reviews_per_month"] = _clip_by_city_quantile(
        combined,
        "reviews_per_month",
        0.99,
    ).round(2)
    combined["number_of_reviews_ltm"] = _clip_by_city_quantile(
        combined,
        "number_of_reviews_ltm",
        0.99,
    ).round(0).astype(int)

    combined["last_review_year"] = combined["last_review"].dt.year.astype("Int64")
    combined = combined.sort_values(
        by=["city", "neighbourhood", "source_listing_id"],
        kind="stable",
    ).reset_index(drop=True)

    return combined[OUTPUT_COLUMNS]


def save_combined_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    source_files: Mapping[str, Path] | None = None,
) -> Path:
    """Generate and save the cleaned comparison dataset."""
    cleaned_dataset = build_combined_dataset(source_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataset.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def _impute_price(dataset: pd.DataFrame) -> pd.Series:
    neighbourhood_median = dataset.groupby(["city", "neighbourhood"])["price"].transform("median")
    city_median = dataset.groupby("city")["price"].transform("median")

    return dataset["price"].fillna(neighbourhood_median).fillna(city_median)


def _clip_by_city_quantile(dataset: pd.DataFrame, column: str, quantile: float) -> pd.Series:
    upper_bounds = dataset.groupby("city")[column].transform(
        lambda values: values.max() if values.count() < 25 else values.quantile(quantile),
    )
    return dataset[column].clip(upper=upper_bounds)
