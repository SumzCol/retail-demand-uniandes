from retail_demand_uniandes.pipelines.modelo_xgboost import create_pipeline as create_xgb_pipeline

def register_pipelines():
    return {
        "xgb": create_xgb_pipeline(),
        "__default__": create_xgb_pipeline(),
    }