import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from colorama import init, Fore, Style

init()

# Cargar el dataset digits
digits = load_digits()
data = digits.images  # Matrices 8x8 con valores de intensidad
target = digits.target  # Etiquetas (0 al 9)

# ===============================
# FUNCIONES DE UTILIDAD
# ===============================

# Calcula la imagen promedio para cada dígito (0-9)
def generar_promedios_digitos():
    promedios = {i: np.zeros((8, 8)) for i in range(10)}
    conteos = {i: 0 for i in range(10)}
    for imagen, digito in zip(data, target):
        promedios[digito] += imagen
        conteos[digito] += 1
    for digito in range(10):
        if conteos[digito] > 0:
            promedios[digito] /= conteos[digito]
    return promedios

# Procesa una imagen externa para que tenga el mismo formato que el dataset
def procesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None or imagen.size == 0:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
    matriz = cv2.resize(imagen, (8, 8))
    matriz = 255 - matriz  # Inversión: fondo blanco y dígito negro
    matriz = matriz / 255 * 16  # Escalado a rango del dataset
    return matriz

# Imprime la matriz en consola con colores para facilitar visualización
def imprimir_matriz(matriz):
    for fila in matriz:
        for valor in fila:
            color = Fore.GREEN if valor > 0 else ''
            print(color + f"{valor:.1f}" + Style.RESET_ALL, end="\t")
        print()

# Retorna las 3 imágenes más cercanas (por distancia euclidiana)
def tres_mas_parecidos(matriz_entrada):
    distancias = []
    for imagen, etiqueta in zip(data, target):
        distancia = np.sqrt(np.sum((matriz_entrada - imagen) ** 2))
        distancias.append((distancia, etiqueta))
    distancias.sort(key=lambda x: x[0])
    return distancias[:3]

# Clasifica según mayoría de votos entre los 3 más cercanos
def clasificacion_version1(tres_cercanos):
    etiquetas = [etq for _, etq in tres_cercanos]
    if etiquetas.count(etiquetas[0]) >= 2:
        return etiquetas[0]
    elif etiquetas.count(etiquetas[1]) >= 2:
        return etiquetas[1]
    else:
        return etiquetas[0]

# Clasifica comparando contra promedios usando distancia euclidiana
def clasificar_digito(matriz_entrada, promedios):
    mejor_digito = None
    menor_distancia = float('inf')
    for digito, promedio in promedios.items():
        distancia = np.sqrt(np.sum((matriz_entrada - promedio) ** 2))
        if distancia < menor_distancia:
            menor_distancia = distancia
            mejor_digito = digito
    return mejor_digito, menor_distancia

# ===============================
# RED NEURONAL LINEAL (VERSIÓN 3)
# ===============================

# Codifica etiquetas a vectores one-hot 
def to_one_hot(y, num_classes=10):
    m = len(y)
    one_hot = np.zeros((m, num_classes))
    for i in range(m):
        one_hot[i, y[i]] = 1
    return one_hot

# Red neuronal sin activaciones: solo capa lineal
# Entrena con MSE y descenso de gradiente
def entrenar_red_lineal():
    X = digits.data / 16.0
    y = digits.target
    y_encoded = to_one_hot(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    W = np.random.randn(64, 10) * 0.01
    b = np.zeros((1, 10))
    lr = 0.1
    epochs = 300

    for epoch in range(epochs):
        Z = X_train @ W + b
        error = Z - y_train
        loss = np.mean(error ** 2)
        dW = X_train.T @ error / len(X_train)
        db = np.mean(error, axis=0, keepdims=True)
        W -= lr * dW
        b -= lr * db
        if epoch % 50 == 0:
            print(f"[V4] Epoch {epoch}: pérdida (MSE) = {loss:.4f}")

    # Clase para predecir con el modelo entrenado
    class ModeloLineal:
        def predict(self, X_input):
            X_input = X_input / 16.0
            Z = X_input @ W + b
            return np.argmax(Z, axis=1)
    return ModeloLineal()

# ===============================
# COMPARACIÓN DE MÉTODOS
# ===============================

# Compara la precisión de las versiones 1, 2 y 4
# con las primeras n imágenes del dataset
def comparar_metodos(n=100):
    promedios = generar_promedios_digitos()
    modelo_lineal = entrenar_red_lineal()

    aciertos_v1 = aciertos_v2 = aciertos_v4 = 0

    for i in range(n):
        matriz = data[i]
        real = target[i]
        flatten = matriz.flatten().reshape(1, -1)

        tres_cercanos = tres_mas_parecidos(matriz)
        pred_v1 = clasificacion_version1(tres_cercanos)
        if pred_v1 == real: aciertos_v1 += 1

        pred_v2, _ = clasificar_digito(matriz, promedios)
        if pred_v2 == real: aciertos_v2 += 1

        pred_v4 = modelo_lineal.predict(flatten)[0]
        if pred_v4 == real: aciertos_v4 += 1

    print("\n--- COMPARACIÓN FINAL ---")
    print(f"Versión 1 (3 vecinos): {aciertos_v1}/{n} = {aciertos_v1/n:.2%}")
    print(f"Versión 2 (promedios): {aciertos_v2}/{n} = {aciertos_v2/n:.2%}")
    print(f"Versión 4 (Red lineal sin activación): {aciertos_v4}/{n} = {aciertos_v4/n:.2%}")

# ===============================
# MAIN
# ===============================

# Función principal: carga la imagen, aplica los métodos y muestra resultados
def main():
    promedios = generar_promedios_digitos()
    modelo_lineal = entrenar_red_lineal()

    try:
        ruta = "un_numero_nuevo.png"
        matriz_entrada = procesar_imagen(ruta)
        print("\nMatriz procesada del número ingresado:")
        imprimir_matriz(matriz_entrada)

        tres_cercanos = tres_mas_parecidos(matriz_entrada)
        digito_v1 = clasificacion_version1(tres_cercanos)
        print(f"\nSoy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {Fore.CYAN}{digito_v1}{Style.RESET_ALL}")

        digito_v2, dist = clasificar_digito(matriz_entrada, promedios)
        print(f"\nSoy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {Fore.YELLOW}{digito_v2}{Style.RESET_ALL}")

        flat = matriz_entrada.flatten().reshape(1, -1)
        pred_v4 = modelo_lineal.predict(flat)[0]
        print(f"\nSoy la inteligencia artificial versión 3, y detecto que el número ingresado es: {Fore.MAGENTA}{pred_v4}{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")

    print("\n--- Comparación general con 100 ejemplos ---")
    comparar_metodos(100)

main()
