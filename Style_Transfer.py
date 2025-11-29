import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# --- PARTE 1: Función para calcular la Matriz de Gram ---
def calcular_matriz_gram(imagen_tensor):
    """
    Esta función demuestra la operación matricial clave.
    La Matriz de Gram es el resultado del producto punto entre 
    los vectores de características de la imagen.
    Fórmula: G = F * F_transpuesta
    """
    # Se obtiene la forma de la matriz de la imagen
    # (Batch, Altura, Anchura, Canales)
    result = tf.linalg.einsum('bijc,bijd->bcd', imagen_tensor, imagen_tensor)
    input_shape = tf.shape(imagen_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

print("--- INICIANDO PROCESO DE MATRICES ---")
print("Cargando modelo de IA...")

# Se usa un modelo pre-entrenado de Google para la extracción de características
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Función para cargar y procesar imágenes (convertirlas a matrices numéricas)
def cargar_imagen(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [384, 384]) # Se redimensiona para rapidez
    img = img[tf.newaxis, :] # Se agrega dimensión batch
    return img

# --- PARTE 2: EJECUCIÓN ---

# 1. Aquí se Cargan las matrices de las imágenes
print("Cargando imágenes...")
content_image = cargar_imagen('Felis_Catus.jpg') # Aqui va la foto
style_image = cargar_imagen('Noche_Estrellada.jpg') # Aqui va el estilo que se le aplica a la foto

# 2. Demostración Matemática
# Aquí se muestra que la IA ve la imagen como números
gram_matrix = calcular_matriz_gram(style_image)
print(f"Dimensiones de la Matriz de Gram calculada: {gram_matrix.shape}")
print("La Matriz de Gram captura la 'textura' matemática del estilo.")

# 3. Aquí se aplica la Transferencia de Estilo
print("Realizando operaciones matriciales de fusión...")
outputs = hub_model(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# 4. Aquí se guarda y se Muestra el resultado
print("¡Imagen generada!")
tf.keras.preprocessing.image.save_img('resultado_final.jpg', stylized_image[0])

# Aquí se muestra un poco de la matriz resultante
print("\nEjemplo de los valores matriciales de la imagen final (Primeros 3 píxeles):")
print(stylized_image[0][0][:3].numpy())