# Librerías necesarias para el proyecto
import cv2                          
import numpy as np                 
import matplotlib.pyplot as plt    
from sklearn.datasets import load_digits         
from sklearn.neural_network import MLPClassifier # Red neuronal multicapa
from sklearn.model_selection import train_test_split  # Para dividir el dataset
from sklearn.metrics import accuracy_score       # Para medir precisión
from colorama import init, Fore, Style            

init()  # Inicializa colorama para usar colores en consola

# --- Cargar el dataset digits ---
digits = load_digits()
data = digits.images     # Imágenes de 8x8 pixeles (0-16)
target = digits.target   # Etiquetas (números reales 0–9)

# --- Mostrar información del dataset ---
print(f"Claves del dataset: {digits.keys()}")
print("Ejemplo de imagen (índice 1000):")
print(digits["images"][1000])  
plt.imshow(digits['images'][1000], cmap="gray")  
plt.title(f'Dígito mostrado: {digits["target"][1000]}')
plt.show()
print("Target del ejemplo:", digits['target'][1000])
print("Nombres de los targets:", digits['target_names'])
print("Descripción del dataset:\n", digits["DESCR"][:500], "...")  

# --- Generar imágenes promedio por dígito (0 al 9) ---
def generar_promedios_digitos():
    promedios = {i: np.zeros((8, 8)) for i in range(10)}  # Diccionario de matrices 8x8 por cada dígito
    conteos = {i: 0 for i in range(10)}                   # Contador de ocurrencias por dígito

    for imagen, digito in zip(data, target):
        promedios[digito] += imagen
        conteos[digito] += 1

    # Calcular el promedio dividiendo entre la cantidad de imágenes de cada clase
    for digito in range(10):
        if conteos[digito] > 0:
            promedios[digito] /= conteos[digito]

    # Mostrar los promedios visualmente
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(promedios[i], cmap='gray', interpolation='nearest')
        ax.set_title(f'Dígito {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return promedios

# --- Procesar imagen externa 8x8 para convertirla al mismo formato que el dataset ---
def procesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {ruta_imagen}")
    if imagen.size == 0:
        raise ValueError("La imagen está vacía o corrupta.")

    matriz_reducida = cv2.resize(imagen, (8, 8))      # Redimensionar
    matriz_reducida = 255 - matriz_reducida           # Invertir colores: fondo blanco, dígito negro
    matriz_reducida = matriz_reducida / 255 * 16      # Escalar a rango 0-16
    return matriz_reducida

# --- Clasificar un dígito comparando con los promedios ---
def clasificar_digito(matriz_entrada, promedios):
    mejor_digito = None
    menor_distancia = float('inf')

    for digito, promedio in promedios.items():
        # Distancia euclidiana entre la imagen ingresada y el promedio
        distancia = np.sqrt(np.sum((matriz_entrada - promedio) ** 2))
        if distancia < menor_distancia:
            menor_distancia = distancia
            mejor_digito = digito

    return mejor_digito, menor_distancia

# --- Imprimir la matriz con formato de colores en consola ---
def imprimir_matriz(matriz):
    for fila in matriz:
        for valor in fila:
            color = Fore.GREEN if valor > 0 else ''
            print(color + f"{valor:.1f}" + Style.RESET_ALL, end="\t")
        print()

# --- Buscar los 3 dígitos más parecidos (mínima distancia euclidiana) ---
def tres_mas_parecidos(matriz_entrada):
    distancias = []
    for imagen, etiqueta in zip(data, target):
        distancia = np.sqrt(np.sum((matriz_entrada - imagen) ** 2))
        distancias.append((distancia, etiqueta))
    distancias.sort(key=lambda x: x[0])  # Ordenar de menor a mayor distancia
    return distancias[:3]

# --- Clasificación versión 1: votación entre los 3 más parecidos ---
def clasificacion_version1(tres_cercanos):
    etiquetas = [etq for _, etq in tres_cercanos]
    # Si al menos 2 etiquetas coinciden, esa es la predicción
    if etiquetas.count(etiquetas[0]) >= 2:
        return etiquetas[0]
    elif etiquetas.count(etiquetas[1]) >= 2:
        return etiquetas[1]
    else:
        return etiquetas[0]  # En caso de 3 distintos, tomar el primero (se puede mejorar)

# --- Entrenar red neuronal artificial (versión 3) ---
def entrenar_red_neuronal():
    X = np.array([img.flatten() for img in data])  # Aplana las matrices 8x8 → vectores de 64 features
    y = target

    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=98)

    # Crear y entrenar la red neuronal multicapa
    modelo = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, alpha=1e-4, solver='adam', random_state=1)
    modelo.fit(X_train, y_train)

    # Evaluar precisión
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    return modelo

# --- Comparar las 3 versiones de clasificación en las primeras N imágenes del dataset ---
def comparar_metodos(n=100):
    promedios = generar_promedios_digitos()
    modelo_rna = entrenar_red_neuronal()

    aciertos_v1 = 0
    aciertos_v2 = 0
    aciertos_v3 = 0

    for i in range(n):
        matriz = data[i]
        real = target[i]
        flatten = matriz.flatten().reshape(1, -1)

        # Versión 1: vecinos
        tres_cercanos = tres_mas_parecidos(matriz)
        pred_v1 = clasificacion_version1(tres_cercanos)

        # Versión 2: promedios
        pred_v2, _ = clasificar_digito(matriz, promedios)

        # Versión 3: red neuronal
        pred_v3 = modelo_rna.predict(flatten)[0]

        # Comparar con etiqueta real
        if pred_v1 == real: aciertos_v1 += 1
        if pred_v2 == real: aciertos_v2 += 1
        if pred_v3 == real: aciertos_v3 += 1

    # Mostrar resultados
    print("\n--- COMPARACIÓN FINAL ---")
    print(f"Versión 1 (3 vecinos): {aciertos_v1}/{n} = {aciertos_v1/n:.2%}")
    print(f"Versión 2 (promedios): {aciertos_v2}/{n} = {aciertos_v2/n:.2%}")
    print(f"Versión 3 (RNA): {aciertos_v3}/{n} = {aciertos_v3/n:.2%}")

# --- Función principal del programa ---
def main():
    promedios = generar_promedios_digitos()

    try:
        ruta_imagen = "un_numero_nuevo.png"  # Ruta a la imagen externa a clasificar
        matriz_entrada = procesar_imagen(ruta_imagen)
        modelo_rna = entrenar_red_neuronal()

        print("\nMatriz procesada del número de entrada:")
        imprimir_matriz(matriz_entrada)

        # Versión 1: vecinos
        tres_cercanos = tres_mas_parecidos(matriz_entrada)
        print("\nLos 3 dígitos más cercanos son (distancia, etiqueta):")
        for distancia, etiqueta in tres_cercanos:
            print(f"Distancia: {distancia:.2f}, Etiqueta: {etiqueta}")
        digito_v1 = clasificacion_version1(tres_cercanos)
        print(f"\nSoy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {Fore.CYAN}{digito_v1}{Style.RESET_ALL} (versión 1)")

        # Versión 2: promedios
        digito_v2, distancia = clasificar_digito(matriz_entrada, promedios)
        print(f"\nSoy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {Fore.YELLOW}{digito_v2}{Style.RESET_ALL}")
        print(f"Distancia al patrón promedio: {distancia:.2f}")

        # Versión 3: red neuronal
        nueva_imagen_flat = matriz_entrada.flatten().reshape(1, -1)
        pred_rna = modelo_rna.predict(nueva_imagen_flat)[0]
        print(f"\nSoy la inteligencia artificial versión 3 (RNA), y detecto que el número ingresado es: {Fore.MAGENTA}{pred_rna}{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")

    print("\n--- Comparación de métodos con 100 imágenes ---")
    comparar_metodos(100)

main()
