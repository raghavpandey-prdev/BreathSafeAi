"""
City feature helper — shared between training and inference so the exact same
city-bucket encoding is used both times. This is critical: if training and
inference build these columns differently, predictions will be silently wrong.
"""

CITY_BUCKETS = ["Delhi-NCR", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune"]

NCR_CITY_NAMES = {"Delhi", "Noida", "Greater Noida", "Gurugram", "Gurgaon",
                  "Faridabad", "Ghaziabad", "Bahadurgarh", "Sonipat"}


def get_city_bucket(city: str) -> str:
    if city in NCR_CITY_NAMES:
        return "Delhi-NCR"
    if city in CITY_BUCKETS:
        return city
    return "Other"


def add_city_dummy_columns(df):
    """Adds one 0/1 column per known city bucket. Returns (df, list_of_new_column_names)."""
    df = df.copy()
    df["city_bucket"] = df["city"].apply(get_city_bucket)
    dummy_cols = []
    for bucket in CITY_BUCKETS:
        col_name = f"city_is_{bucket.replace('-', '_')}"
        df[col_name] = (df["city_bucket"] == bucket).astype(int)
        dummy_cols.append(col_name)
    return df, dummy_cols