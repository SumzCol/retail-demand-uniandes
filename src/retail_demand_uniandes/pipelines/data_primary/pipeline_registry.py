from kedro.pipeline import Pipeline
from retail_demand_uniandes.pipelines import data_engineering, data_primary

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_engineering.create_pipeline() + data_primary.create_pipeline(),
        "data_engineering": data_engineering.create_pipeline(),
        "data_primary": data_primary.create_pipeline(),
    }