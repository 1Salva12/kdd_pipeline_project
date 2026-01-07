import arff
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Importante: Para que matplotlib no intente abrir ventanas en el servidor
import matplotlib.pyplot as plt
import io
import base64
from django.conf import settings
import os

def get_plot():
    # Función auxiliar para convertir el gráfico en imagen para HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def process_kdd_data():
    path = os.path.join(settings.BASE_DIR, 'datasets', 'KDDTrain+.arff')
    
    with open(path, 'r') as f:
        dataset = arff.load(f)
        attributes = [attr[0] for attr in dataset['attributes']]
        df = pd.DataFrame(dataset["data"], columns=attributes)

    # --- NUEVO: Generar Gráfico de Distribución ---
    plt.figure(figsize=(8, 4))
    df['class'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
    plt.title('Distribución de Tráfico: Normal vs Ataque')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    
    chart = get_plot() # Obtenemos la imagen en base64
    plt.close() # Cerramos el gráfico para liberar memoria

    # Resumen para mostrar en la web
    info = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'class_counts': df['class'].value_counts().to_dict(),
        'sample_data': df.head(10).to_html(classes='table table-striped table-hover'),
        'chart': chart  # <-- Pasamos el gráfico aquí
    }
    
    # ... (El resto de tu lógica de preprocesamiento se mantiene igual)
    # Solo asegúrate de retornar el diccionario 'info' actualizado
    return info