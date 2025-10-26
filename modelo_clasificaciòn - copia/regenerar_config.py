import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

model_save_path = "./modelo_clasificador_final"
df = pd.read_csv("dataset_final_preparado.csv")

categorias = ['Seguridad', 'Educación', 'Medio Ambiente', 'Salud']
label2id = {cat: i for i, cat in enumerate(categorias)}
id2label = {i: cat for i, cat in enumerate(categorias)}

df['label'] = df['Categoría del problema'].map(label2id)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

config_data = {
    'categorias': categorias,
    'label2id': label2id,
    'id2label': {str(k): v for k, v in id2label.items()},
    'total_registros': len(df),
    'registros_entrenamiento': len(train_df),
    'registros_prueba': len(test_df),
    'precision_modelo': 0.0,  # Puedes actualizarlo si tienes el valor real
    'epocas': 5,
    'fecha_entrenamiento': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(model_save_path, "config_categorias.json"), "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)

print("Archivo config_categorias.json generado correctamente.")