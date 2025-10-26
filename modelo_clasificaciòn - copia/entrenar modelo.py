# entrenar_modelo.py

"""
Script para entrenar un modelo de clasificación de comentarios en español utilizando BERT.
Este script debe ejecutarse una sola vez para entrenar el modelo con el dataset completo
y guardar el modelo entrenado junto con su configuración.

Requisitos:
- Archivo 'dataset_final_preparado.csv' en el mismo directorio.
- Paquetes: pandas, torch, transformers, datasets, scikit-learn, numpy, json

Salida:
- Modelo entrenado guardado en './modelo_clasificador_final'
- Reporte de clasificación
- Archivo de configuración 'config_categorias.json'
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import json

# Configuración del entorno para evitar mensajes de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NO_TF"] = "1"

print("="*70)
print("ENTRENAMIENTO DE MODELO - CLASIFICADOR DE COMENTARIOS")
print("="*70)

# Paso 1: Cargar el dataset
print("\nPaso 1: Cargando dataset...")
try:
    df = pd.read_csv('dataset_final_preparado.csv')
    print(f"Dataset cargado: {len(df):,} registros")
except FileNotFoundError:
    print("ERROR: No se encuentra 'dataset_final_preparado.csv'")
    exit()

# Paso 2: Configurar categorías y etiquetas
print("\nPaso 2: Configurando categorías...")
categorias = ['Seguridad', 'Educación', 'Medio Ambiente', 'Salud']
label2id = {cat: i for i, cat in enumerate(categorias)}
id2label = {i: cat for i, cat in enumerate(categorias)}
df['label'] = df['Categoría del problema'].map(label2id)

# Mostrar distribución de clases
print("\nDistribución del dataset:")
for cat in categorias:
    count = len(df[df['Categoría del problema'] == cat])
    porcentaje = (count / len(df)) * 100
    print(f"{cat:20s}: {count:5d} ({porcentaje:5.1f}%)")

# Paso 3: Dividir datos en entrenamiento y prueba
print("\nPaso 3: Dividiendo datos (80% entrenamiento, 20% prueba)...")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Entrenamiento: {len(train_df):,} registros")
print(f"Prueba: {len(test_df):,} registros")

# Paso 4: Cargar tokenizador BERT en español
print("\nPaso 4: Cargando tokenizador BERT en español...")
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
print("Tokenizador cargado")

# Función para tokenizar los textos
def tokenize_function(examples):
    return tokenizer(examples['Comentario'], padding=True, truncation=True, max_length=128)

# Paso 5: Preparar datasets tokenizados
print("\nPaso 5: Preparando datasets tokenizados...")
train_dataset = Dataset.from_pandas(train_df[['Comentario', 'label']])
test_dataset = Dataset.from_pandas(test_df[['Comentario', 'label']])
train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)
print("Datasets preparados")

# Paso 6: Cargar modelo BERT base
print("\nPaso 6: Cargando modelo BERT base...")
model = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)
print("Modelo cargado")

# Paso 7: Configurar parámetros de entrenamiento
print("\nPaso 7: Configurando parámetros de entrenamiento...")
training_args = TrainingArguments(
    output_dir="./modelo_temporal",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=None,
    logging_steps=100,
)

# Función para calcular métricas de evaluación
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Inicializar entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    compute_metrics=compute_metrics,
)

print("Configuración lista")

# Paso 8: Entrenar el modelo
print("\n" + "="*70)
print("INICIANDO ENTRENAMIENTO")
print("="*70 + "\n")
trainer.train()
print("\n" + "="*70)
print("ENTRENAMIENTO COMPLETADO")
print("="*70)

# Paso 9: Guardar modelo entrenado
print("\nPaso 9: Guardando modelo entrenado...")
model_save_path = "./modelo_clasificador_final"
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Modelo guardado en: {model_save_path}/")

# Paso 10: Evaluar precisión del modelo
print("\nPaso 10: Evaluando precisión del modelo...")
predicciones = []
reales = []

for i in range(len(test_df)):
    texto = test_df.iloc[i]['Comentario']
    etiqueta_real = test_df.iloc[i]['label']
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = probs.argmax().item()
    predicciones.append(predicted_id)
    reales.append(etiqueta_real)

# Reporte de clasificación
print("\nREPORTE DE CLASIFICACIÓN:")
print("="*70)
report = classification_report(
    reales,
    predicciones,
    target_names=categorias,
    digits=3
)
print(report)
accuracy = (np.array(predicciones) == np.array(reales)).mean()

# Paso 11: Guardar configuración del modelo
print("\nPaso 11: Guardando configuración...")
config_path = os.path.join(model_save_path, "config_categorias.json")
config_data = {
    'categorias': categorias,
    'label2id': label2id,
    'id2label': {str(k): v for k, v in id2label.items()},
    'total_registros': len(df),
    'registros_entrenamiento': len(train_df),
    'registros_prueba': len(test_df),
    'precision_modelo': float(accuracy),
    'epocas': 5,
    'fecha_entrenamiento': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)
print("Configuración guardada")

# Resumen final
print("\n" + "="*70)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"\nRESULTADOS:")
print(f"Precisión del modelo: {accuracy:.1%}")
print(f"Registros entrenados: {len(train_df):,}")
print(f"Registros de prueba: {len(test_df):,}")
print(f"\nARCHIVOS GENERADOS:")
print(f"{model_save_path}/")
print(f"  ├── pytorch_model.bin")
print(f"  ├── config.json")
print(f"  ├── tokenizer_config.json")
print(f"  ├── vocab.txt")
print(f"  └── config_categorias.json")
print(f"\nSIGUIENTE PASO:")
print(f"Ejecuta: python usar_modelo.py")
print("="*70 + "\n")
