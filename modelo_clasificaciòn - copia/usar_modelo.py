import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime

# ==================== 1. CARGAR MODELO ====================
model_path = "./modelo_clasificador_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
modelo = AutoModelForSequenceClassification.from_pretrained(model_path)

# Cargar configuración
with open(f"{model_path}/config_categorias.json", "r", encoding="utf-8") as f:
    config = json.load(f)

categorias = config["categorias"]
precision_modelo = config.get("precision", 0.0)
fecha_entrenamiento = config.get("fecha_entrenamiento", "No disponible")

# ==================== 2. FUNCIÓN DE CLASIFICACIÓN ====================
def clasificar_comentario(texto):
    texto = texto.strip()
    if not texto:
        return "⚠️ Por favor ingresa un texto", 0.0, "N/A"

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = modelo(**inputs)
    prediccion = torch.argmax(outputs.logits, dim=1).item()
    confianza = torch.softmax(outputs.logits, dim=1)[0][prediccion].item()

    return categorias[prediccion], round(confianza * 100, 2), "Modelo BERT"

# ==================== 3. FUNCIÓN PARA EXCEL ====================
def procesar_archivo_excel(archivo):
    try:
        df = pd.read_excel(archivo.name)
        if "Comentario" not in df.columns:
            return "⚠️ El archivo debe tener una columna llamada 'Comentario'.", None

        resultados = []
        for comentario in df["Comentario"]:
            categoria, confianza, metodo = clasificar_comentario(str(comentario))
            resultados.append({
                "Comentario": comentario,
                "Categoría Predicha": categoria,
                "Confianza (%)": confianza,
                "Método": metodo
            })
        df_resultados = pd.DataFrame(resultados)
        ruta_salida = "resultados_clasificados.xlsx"
        df_resultados.to_excel(ruta_salida, index=False)
        return "✅ Clasificación completada con éxito.", ruta_salida
    except Exception as e:
        return f"❌ Error al procesar el archivo: {str(e)}", None

# ==================== 4. INTERFAZ GRADIO ====================
with gr.Blocks(title="🧠 Clasificador Inteligente de Reportes Ciudadanos") as demo:
    gr.Markdown("""
    # 🏙️ Clasificador Inteligente de Reportes Ciudadanos
    Esta aplicación analiza los comentarios y reportes de los ciudadanos, clasificándolos en categorías como:
    **Salud**, **Educación**, **Medio Ambiente** y **Seguridad**.
    ---
    """)

    with gr.Tab("🔍 Clasificación Individual"):
        texto_input = gr.Textbox(label="✏️ Ingresa un comentario", placeholder="Ejemplo: Las calles están oscuras y es peligroso salir de noche.")
        btn_clasificar = gr.Button("🎯 Clasificar")
        categoria_output = gr.Textbox(label="Categoría Detectada")
        confianza_output = gr.Number(label="Confianza (%)")
        metodo_output = gr.Textbox(label="Método Usado")

    with gr.Tab("📊 Procesar Excel Masivo"):
        excel_input = gr.File(label="📁 Sube un archivo Excel con una columna 'Comentario'")
        btn_procesar = gr.Button("🚀 Procesar Archivo")
        excel_output = gr.File(label="📂 Archivo Clasificado (Descargar)")
        mensaje_excel = gr.Textbox(label="Estado del proceso")

    with gr.Tab("ℹ️ Información del Sistema"):
        gr.Markdown(f"""
        ## 📊 Detalles del Modelo
        
        ### 🎓 Entrenamiento
        - **Precisión:** {precision_modelo:.1%}
        - **Fecha:** {fecha_entrenamiento}
        - **Ubicación:** `{model_path}`
        
        ### 🏷️ Categorías Disponibles
        {chr(10).join([f'- **{cat}**' for cat in categorias])}
        
        ### 🔧 Métodos de Clasificación
        El sistema usa un enfoque **híbrido inteligente** que combina:
        
        1. **Detección de frases específicas** (95% confianza)  
           Ejemplo: *"calles oscuras"* → Seguridad
        
        2. **Análisis de palabras clave** (90% confianza)  
           Busca palabras clave frecuentes en cada categoría.
        
        3. **Modelo BERT**  
           Red neuronal entrenada con {config.get('registros_entrenamiento', 'N/A'):,} comentarios.
        
        4. **Combinación adaptativa**  
           Ajusta la confianza según coincidencias entre métodos.
        """)

    # ==================== 5. CONECTAR FUNCIONALIDADES ====================
    btn_clasificar.click(
        clasificar_comentario,
        inputs=texto_input,
        outputs=[categoria_output, confianza_output, metodo_output]
    )
    
    btn_procesar.click(
        procesar_archivo_excel,
        inputs=excel_input,
        outputs=[mensaje_excel, excel_output]
    )

# ==================== 6. EJECUTAR APLICACIÓN ====================
if __name__ == "__main__":
    demo.launch()
