import cv2
import numpy as np
from sklearn.datasets import load_digits
from colorama import init, Fore, Style

init()

def generar_promedios_digitos():
    digits = load_digits()
    data = digits.images
    target = digits.target
    
    promedios = {i: np.zeros((8, 8)) for i in range(10)}
    conteos = {i: 0 for i in range(10)}
    
    for imagen, digito in zip(data, target):
        promedios[digito] += imagen
        conteos[digito] += 1
    
    for digito in range(10):
        if conteos[digito] > 0:
            promedios[digito] /= conteos[digito]
    
    return promedios

def procesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {ruta_imagen}")
    
    matriz_reducida = cv2.resize(imagen, (8, 8))
    matriz_reducida = 255 - matriz_reducida
    matriz_reducida = matriz_reducida / 255 * 16
    
    return matriz_reducida

def clasificar_digito(matriz_entrada, promedios):
    mejor_digito = None
    menor_distancia = float('inf')
    
    for digito, promedio in promedios.items():
        distancia = np.sqrt(np.sum((matriz_entrada - promedio) ** 2))
        
        if distancia < menor_distancia:
            menor_distancia = distancia
            mejor_digito = digito
    
    return mejor_digito, menor_distancia

def imprimir_matriz(matriz):
    for fila in matriz:
        for valor in fila:
            if valor > 0:
                print(Fore.GREEN + f"{valor:.1f}" + Style.RESET_ALL, end="\t")
            else:
                print(f"{valor:.1f}", end="\t")
        print()



def tres_mas_parecidos(matriz_entrada):
    digits = load_digits()
    data = digits.images
    target = digits.target
    distancias = []

    for imagen, etiqueta in zip(data, target):
        distancia = np.sqrt(np.sum((matriz_entrada - imagen) ** 2))
        distancias.append((distancia, etiqueta))

    distancias.sort(key=lambda x: x[0])
    tres_cercanos = distancias[:3]
    return tres_cercanos

def clasificacion_version1(tres_cercanos):
    etiquetas = [etq for _, etq in tres_cercanos]
    if etiquetas.count(etiquetas[0]) >= 2:
        return etiquetas[0]
    elif etiquetas.count(etiquetas[1]) >= 2:
        return etiquetas[1]
    else:
        return etiquetas[0]

def comparar_metodos(n=100):
    digits = load_digits()
    data = digits.images
    target = digits.target
    promedios = generar_promedios_digitos()
    aciertos_v1 = 0
    aciertos_v2 = 0

    for i in range(n):
        matriz = data[i]
        real = target[i]
        tres_cercanos = tres_mas_parecidos(matriz)
        pred_v1 = clasificacion_version1(tres_cercanos)
        pred_v2, _ = clasificar_digito(matriz, promedios)
        if pred_v1 == real:
            aciertos_v1 += 1
        if pred_v2 == real:
            aciertos_v2 += 1

    print(f"Precisión versión 1 (3 vecinos): {aciertos_v1}/{n} = {aciertos_v1/n:.2%}")
    print(f"Precisión versión 2 (promedios): {aciertos_v2}/{n} = {aciertos_v2/n:.2%}")

def main():
    promedios = generar_promedios_digitos()
    
    try:
        ruta_imagen = "un_numero_nuevo.png" 
        matriz_entrada = procesar_imagen(ruta_imagen)
        
        print("\nMatriz procesada del número de entrada:")
        imprimir_matriz(matriz_entrada)
        
        # Versión 1: Buscar los 3 más cercanos en el dataset
        tres_cercanos = tres_mas_parecidos(matriz_entrada)
        print("\nLos 3 dígitos más cercanos son (distancia, etiqueta):")
        for distancia, etiqueta in tres_cercanos:
            print(f"Distancia: {distancia:.2f}, Etiqueta: {etiqueta}")
        
        digito_v1 = clasificacion_version1(tres_cercanos)
        print(f"\nSoy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {Fore.CYAN}{digito_v1}{Style.RESET_ALL} (versión 1)")
        
        # Versión 2: Clasificación por promedios
        digito_v2, distancia = clasificar_digito(matriz_entrada, promedios)
        print(f"\nSoy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {Fore.YELLOW}{digito_v2}{Style.RESET_ALL}")
        print(f"Distancia al patrón promedio: {distancia:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

main()