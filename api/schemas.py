"""Request and response models for the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class ShipmentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Warehouse_block: str = Field(min_length=1)
    Mode_of_Shipment: str = Field(min_length=1)
    Customer_care_calls: float = Field(ge=0)
    Customer_rating: float = Field(ge=1, le=5)
    Cost_of_the_Product: float = Field(ge=0)
    Prior_purchases: float = Field(ge=0)
    Product_importance: str = Field(min_length=1)
    Gender: str = Field(min_length=1)
    Discount_offered: float = Field(ge=0)
    Weight_in_gms: float = Field(ge=0)


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    delivery_status: str
    late_probability: float = Field(ge=0, le=1)
