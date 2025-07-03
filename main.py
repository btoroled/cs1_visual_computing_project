import numpy as np

# Datos de entrenamiento
X = np.array([1, 2, 3, 4, 5])  # Entradas
Y = 0.5 * X  # Salidas reales

# Inicialización del peso
w = np.random.rand()

# Parámetros de entrenamiento
epochs = 10000
learning_rate = 0.0001

# Entrenamiento
for epoch in range(epochs):
    for x, y_real in zip(X, Y):
        y_pred = w * x
        error = y_pred - y_real
        loss = 0.5 * (error ** 2) # no se necesita en el código

        # Gradiente
        gradient = error * x

        # Actualización del peso
        w -= learning_rate * gradient

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Weight {w:.4f}')

# Resultado
print(f'Final weight: {w:.4f}')





