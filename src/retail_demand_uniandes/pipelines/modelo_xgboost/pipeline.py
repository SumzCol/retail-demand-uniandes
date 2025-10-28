from kedro.pipeline import Pipeline, node
from .nodes import preprocess_and_split, train_predict_evaluate_xgboost, concat_dicttodf

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess_and_split,
            inputs=["olist_consolidated_dataset", "params:category_col", "params:date_col", "params:target_col", "params:test_days"],
            outputs=["traindict", "testdict", "intermediate_table"],
            name="preprocess_and_split_node",
),
        node(
            func=concat_dicttodf,
            inputs="train_dict",
            outputs="train_concat",
            name="train_concat_node"
        ),
        node(
            func=concat_dicttodf,
            inputs="test_dict",
            outputs="test_concat",
            name="test_concat_node"
        ),
        node(
            func=train_predict_evaluate_xgboost,
            inputs=["train_dict", "test_dict", "params:xgb_param_distributions", "params:xgb_random_state"],
            outputs="metrics_by_category",
            name="train_predict_evaluate_xgboost_node"
        ),
    ])
