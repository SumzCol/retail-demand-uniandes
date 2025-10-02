from kedro.pipeline import Pipeline
from retail_demand_uniandes.pipelines.modelo_sarimax import pipeline as modelo_sarimax_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    modelo_sarimax = modelo_sarimax_pipeline.create_pipeline()
    return {
        "modelo_sarimax": modelo_sarimax,
        "__default__": modelo_sarimax,  # opcional: que sea la default
    }