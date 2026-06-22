import pandas as pd

# =========================================================
# DATA PREPARATION
# =========================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def prepare_spend_df(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df = df.fillna(0)
    return df


def prepare_budget_df(df: pd.DataFrame) -> pd.DataFrame:

    df = clean_columns(df)
    df = df.fillna(0)

    # Fix date parsing for format like 01-04-2025
    if "Date" in df.columns:

     if pd.api.types.is_numeric_dtype(df["Date"]):

        df["Date"] = pd.to_datetime(
            df["Date"],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    else:

        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

    print("\nDate conversion check:")
    print(df["Date"].head())

    budget_parts = [
        "MNP-Total",
        "Electricity",
        "PANTRY",
        "HOUSEKEEPING",
        "STATIONARY",
        "RENTAL / AMENITIES",
        "R&M",
        "IT",
        "OTHERS",
        "PARKING"
    ]

    existing_parts = [
        c for c in budget_parts
        if c in df.columns
    ]

    print("\n========== PREPARE BUDGET ==========")
    print("Matched columns:")
    print(existing_parts)

    if existing_parts:

        for col in existing_parts:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

        df["Total_Budget"] = df[
            existing_parts
        ].sum(axis=1)

        print("Total_Budget created successfully")

        print(
            df[
                ["Centre Code", "Date", "Total_Budget"]
            ].head()
        )

    else:

        print("WARNING: Total_Budget NOT CREATED")

    print("====================================")

    return df


# =========================================================
# GENERIC FILTER ENGINE
# =========================================================

def apply_filters(
    df: pd.DataFrame,
    filters: dict | None = None
) -> pd.DataFrame:

    if not filters:
        return df

    filtered_df = df.copy()

    print("\n========== APPLY FILTERS ==========")
    print("Filters:", filters)
    print("Starting rows:", len(filtered_df))

    for column, value in filters.items():

        if value is None or value == "":
            continue

        if column not in filtered_df.columns:

            print(
                f"Column not found: {column}"
            )

            continue

        print(
            f"\nFiltering {column} = {value}"
        )

        # Date columns
        if pd.api.types.is_datetime64_any_dtype(
            filtered_df[column]
        ):

            if isinstance(value, int):

                before = len(filtered_df)

                filtered_df = filtered_df[
                    filtered_df[column]
                    .dt.year
                    == value
                ]

                print(
                    f"Year filter: "
                    f"{before} -> {len(filtered_df)}"
                )

            else:

                value_dt = pd.to_datetime(
                    value,
                    errors="coerce"
                )

                if pd.notna(value_dt):

                    before = len(filtered_df)

                    filtered_df = filtered_df[
                        filtered_df[column]
                        .dt.year
                        == value_dt.year
                    ]

                    print(
                        f"Date filter: "
                        f"{before} -> {len(filtered_df)}"
                    )

        # Text columns
        else:

            before = len(filtered_df)

            filtered_df = filtered_df[
                filtered_df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(value)
                .strip()
                .lower()
            ]

            print(
                f"Text filter: "
                f"{before} -> {len(filtered_df)}"
            )

    print(
        "\nRows after filtering:",
        len(filtered_df)
    )

    print("===================================")

    return filtered_df


# =========================================================
# SAFE NUMERIC CONVERTER
# =========================================================

def safe_numeric(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:

    df = df.copy()

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# =========================================================
# GENERIC METRIC CALCULATOR
# =========================================================

def calculate_metric(
    df: pd.DataFrame,
    metric_column: str,
    filters: dict | None = None,
    aggregation: str = "sum"
):

    if metric_column not in df.columns:

        return {
            "error": f"Column not found: {metric_column}"
        }

    df = apply_filters(
        df,
        filters
    )

    df = safe_numeric(
        df,
        metric_column
    )

    series = df[metric_column]

    if aggregation == "sum":
        return series.sum()

    if aggregation == "avg":
        return series.mean()

    if aggregation == "max":
        return series.max()

    if aggregation == "min":
        return series.min()

    if aggregation == "count":
        return series.count()

    return {
        "error": f"Invalid aggregation: {aggregation}"
    }


# =========================================================
# METRIC DEFINITIONS
# =========================================================

METRICS = {
    "cost_per_sq_ft": {
        "source": "spend",
        "column": "[Cost_Per_Sq_FT]",
        "aggregation": "sum"
    },

    "avg_cost_per_sq_ft": {
        "source": "spend",
        "column": "[Cost_Per_Sq_FT]",
        "aggregation": "avg"
    },

    "budget_per_sq_feet": {
        "source": "spend",
        "column": "[Budget_per_sq_feet]",
        "aggregation": "sum"
    },

    "over_spend": {
        "source": "spend",
        "column": "[Over_Spend]",
        "aggregation": "sum"
    },

    "sbua": {
        "source": "spend",
        "column": "[SBUA]",
        "aggregation": "sum"
    },

    "sum_bpcl_cost": {
        "source": "spend",
        "column": "[SumBPCL_Cost]",
        "aggregation": "sum"
    },

    "total_budget": {
        "source": "budget",
        "column": "Total_Budget",
        "aggregation": "sum"
    }
}


# =========================================================
# MAIN ROUTER
# =========================================================

def get_metric(
    data_sources: dict,
    metric: str,
    filters: dict | None = None
):

    metric = metric.strip().lower()

    if metric not in METRICS:

        return {
            "error": f"Unknown metric: {metric}",
            "available_metrics": list(METRICS.keys())
        }

    metric_info = METRICS[metric]

    source_name = metric_info["source"]

    if source_name not in data_sources:

        return {
            "error": f"Missing data source: {source_name}"
        }

    df = data_sources[source_name]

    print("\n========== GET METRIC ==========")
    print("Metric:", metric)
    print("Source:", source_name)
    print("Filters:", filters)

    value = calculate_metric(
        df=df,
        metric_column=metric_info["column"],
        filters=filters,
        aggregation=metric_info["aggregation"]
    )

    print("Calculated value:", value)
    print("================================")

    if isinstance(value, dict) and "error" in value:
        return value

    try:
        value = round(float(value), 2)
    except Exception:
        pass

    return {
        "metric": metric,
        "source": source_name,
        "value": value,
        "filters": filters or {}
    }