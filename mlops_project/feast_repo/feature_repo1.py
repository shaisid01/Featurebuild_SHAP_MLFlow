from feast import Entity, FeatureView, Field
from feast.types import Float32
from feast import FileSource

entity = Entity(
    name="customer",
    join_keys=["ID_code"]
)

data_source = FileSource(
    path="../feature_store/features.parquet",
    timestamp_field="event_timestamp"
)

feature_view = FeatureView(
    name="transaction_features",
    entities=[entity],
    schema=[
        Field(name="row_mean", dtype=Float32),
        Field(name="row_std", dtype=Float32),
    ],
    source=data_source,
)