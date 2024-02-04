import numpy as np
import matplotlib.pyplot as plt

# Definir la función de activación de escalón
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Generar valores de entrada
x = np.linspace(-5, 5, 100)  # Generar 100 puntos entre -5 y 5

# Calcular los valores de salida utilizando la función de activación de escalón
y = step_function(x)

# Graficar la función de activación de escalón
plt.plot(x, y)
plt.title('Función de Activación de Escalón')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
