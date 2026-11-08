"""Canonical dataset and model feature definitions."""

TARGET_COLUMN = "Reached.on.Time_Y.N"
ID_COLUMN = "ID"

NUMERIC_FEATURES = (
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms",
)
CATEGORICAL_FEATURES = (
    "Warehouse_block",
    "Mode_of_Shipment",
    "Product_importance",
    "Gender",
)
FEATURE_COLUMNS = (
    "Warehouse_block",
    "Mode_of_Shipment",
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Product_importance",
    "Gender",
    "Discount_offered",
    "Weight_in_gms",
)
REQUIRED_TRAINING_COLUMNS = (ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
