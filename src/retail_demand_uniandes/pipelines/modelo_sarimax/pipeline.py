from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_and_aggregate,
    adf_test_by_category,
    difference_series,
    prepare_exog,
    train_sarimax_by_category,
    summarize_results,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_and_aggregate,
            inputs=dict(df_raw="olist_consolidated_dataset", params="params:modelo_sarimax"),
            outputs="resultadofinal",
            name="clean_and_aggregate",
        ),
        node(
            func=adf_test_by_category,
            inputs=dict(agg="resultadofinal", params="params:modelo_sarimax"),
            outputs=["resultados_adf", "productos_no_estacionarios"],
            name="adf_test_by_category",
        ),
        node(
            func=difference_series,
            inputs=dict(agg="resultadofinal", non_stationary="productos_no_estacionarios", params="params:modelo_sarimax"),
            outputs=["series_diferenciadas", "d_map"],
            name="difference_series",
        ),
        node(
            func=prepare_exog,
            inputs=dict(agg_diff="series_diferenciadas", params="params:modelo_sarimax"),
            outputs="exog_dict",
            name="prepare_exog",
        ),
        node(
            func=train_sarimax_by_category,
            inputs=dict(agg_diff="series_diferenciadas", d_map="d_map", exog_dict="exog_dict", params="params:modelo_sarimax"),
            outputs="resultados_modelos_sarimax",
            name="train_sarimax_by_category",
        ),
        node(
            func=summarize_results,
            inputs=dict(results="resultados_modelos_sarimax", params="params:modelo_sarimax"),
            outputs="resumen_top_modelos",
            name="summarize_results",
        ),
    ])