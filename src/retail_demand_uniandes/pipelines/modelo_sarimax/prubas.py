import pandas as pd

df_resultados = pd.read_parquet("data/04_feature/resultados_modelos_sarimax.parquet")
df_resumen = pd.read_parquet("data/05_model_input/resumen_top_modelos.parquet")


print(df_resultados.head(70))
print(df_resumen)
print(df_resultados.shape)

