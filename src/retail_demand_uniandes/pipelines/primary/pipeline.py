from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_primary_dataset, calculate_dias_venta, filter_productos_mas_45

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_primary_dataset,
            inputs="olist_consolidated_dataset",
            outputs="primary_dataset",
            name="build_primary_dataset_node"
        ),
        node(
            func=calculate_dias_venta,
            inputs="primary_dataset",
            outputs="dataset_con_dias_venta",
            name="calculate_dias_venta_node"
        ),
        node(
            func=filter_productos_mas_45,
            inputs="dataset_con_dias_venta",
            outputs="productos_filtrados",
            name="filter_productos_mas_45_node"
        ),
    ])
