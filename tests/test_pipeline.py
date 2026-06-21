from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".tmp_tests"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rental_market_analysis.pipeline import build_combined_dataset


def _raw_row(
    listing_id: int,
    *,
    city_group: str | None,
    neighbourhood: str,
    price: float | None,
    minimum_nights: int,
    number_of_reviews: int,
    last_review: str | None,
    reviews_per_month: float | None,
    number_of_reviews_ltm: int,
) -> dict[str, object]:
    return {
        "id": listing_id,
        "name": f"Listing {listing_id}",
        "host_id": listing_id * 100,
        "host_name": f"Host {listing_id}",
        "neighbourhood_group": city_group,
        "neighbourhood": neighbourhood,
        "latitude": 45.0,
        "longitude": 9.0,
        "room_type": "Entire home/apt",
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "last_review": last_review,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": 1,
        "availability_365": 180,
        "number_of_reviews_ltm": number_of_reviews_ltm,
        "license": None,
    }


def _prepare_case_dir(case_name: str) -> Path:
    case_dir = TEST_TMP_ROOT / case_name
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


class BuildCombinedDatasetTests(unittest.TestCase):
    def test_pipeline_imputes_price_and_preserves_review_precision(self) -> None:
        tmp_path = _prepare_case_dir("case_price_imputation")
        zurich_path = tmp_path / "zurich.csv"
        milan_path = tmp_path / "milan.csv"

        pd.DataFrame(
            [
                _raw_row(
                    1,
                    city_group="Kreis 1",
                    neighbourhood="Centro",
                    price=120.0,
                    minimum_nights=3,
                    number_of_reviews=10,
                    last_review="2024-01-01",
                    reviews_per_month=2.75,
                    number_of_reviews_ltm=20,
                ),
                _raw_row(
                    2,
                    city_group="Kreis 1",
                    neighbourhood="Centro",
                    price=None,
                    minimum_nights=7,
                    number_of_reviews=4,
                    last_review="2024-02-01",
                    reviews_per_month=1.25,
                    number_of_reviews_ltm=10,
                ),
            ]
        ).to_csv(zurich_path, index=False)

        pd.DataFrame(
            [
                _raw_row(
                    3,
                    city_group=None,
                    neighbourhood="Navigli",
                    price=200.0,
                    minimum_nights=5,
                    number_of_reviews=3,
                    last_review="2024-03-10",
                    reviews_per_month=0.5,
                    number_of_reviews_ltm=5,
                )
            ]
        ).to_csv(milan_path, index=False)

        dataset = build_combined_dataset({"Zurich": zurich_path, "Milan": milan_path})

        imputed_row = dataset.loc[dataset["source_listing_id"] == 2].iloc[0]
        reference_row = dataset.loc[dataset["source_listing_id"] == 1].iloc[0]

        self.assertEqual(imputed_row["price"], 120.0)
        self.assertTrue(imputed_row["price_imputed"])
        self.assertEqual(reference_row["reviews_per_month"], 2.75)
        self.assertFalse(reference_row["reviews_per_month_imputed"])

    def test_pipeline_clips_long_stays_and_backfills_missing_review_metrics(self) -> None:
        tmp_path = _prepare_case_dir("case_review_defaults")
        zurich_path = tmp_path / "zurich.csv"
        milan_path = tmp_path / "milan.csv"

        pd.DataFrame(
            [
                _raw_row(
                    10,
                    city_group="Kreis 2",
                    neighbourhood="Seefeld",
                    price=300.0,
                    minimum_nights=800,
                    number_of_reviews=0,
                    last_review=None,
                    reviews_per_month=None,
                    number_of_reviews_ltm=0,
                )
            ]
        ).to_csv(zurich_path, index=False)

        pd.DataFrame(
            [
                _raw_row(
                    11,
                    city_group=None,
                    neighbourhood="Duomo",
                    price=150.0,
                    minimum_nights=2,
                    number_of_reviews=12,
                    last_review="2024-05-05",
                    reviews_per_month=1.5,
                    number_of_reviews_ltm=30,
                )
            ]
        ).to_csv(milan_path, index=False)

        dataset = build_combined_dataset({"Zurich": zurich_path, "Milan": milan_path})
        clipped_row = dataset.loc[dataset["source_listing_id"] == 10].iloc[0]

        self.assertEqual(clipped_row["minimum_nights"], 365)
        self.assertEqual(clipped_row["reviews_per_month"], 0.0)
        self.assertTrue(clipped_row["reviews_per_month_imputed"])
        self.assertTrue(pd.isna(clipped_row["last_review_year"]))


if __name__ == "__main__":
    unittest.main()
