import pandas as pd
import numpy as np

def build_primary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    # Fechas
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "shipping_limit_date",
        "review_creation_date", "review_answer_timestamp"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["order_id", "customer_id"])

    # Relleno numérico
    num_cols_fill0 = ["price", "freight_value", "payment_installments", "payment_value"]
    for col in num_cols_fill0:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Relleno texto
    text_cols_fill = [
        "product_category_name", "product_category_name_english",
        "seller_city", "seller_state", "customer_city", "customer_state"
    ]
    for col in text_cols_fill:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Coordenadas
    coord_cols = ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
    for col in coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=coord_cols)

    # Renombrar columnas
    df = df.rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length"
    })

    # Variable auxiliar
    df["cantidad"] = 1

    return df


def calculate_dias_venta(df: pd.DataFrame) -> pd.DataFrame:
    dias_venta = (
        df.groupby("product_category_name_english")["order_purchase_timestamp"]
        .apply(lambda x: pd.to_datetime(x).dt.date.nunique())
        .reset_index(name="dias_venta")
    )
    return dias_venta


def filter_productos_mas_45(dias_venta: pd.DataFrame) -> pd.DataFrame:
    return dias_venta[dias_venta["dias_venta"] > 45]


def ejemplo_node(data: dict) -> dict:
    """Nodo de prueba que solo imprime y retorna lo mismo."""
    print("🚀 Nodo de prueba ejecutado con éxito")
    return data
