import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk

# Función para asignar pesos aleatorios
def inicializar_pesos(num_features):
    #Semilla que se encarga que se inicialice con el mismo 
    #np.random.seed(0)
    return np.random.rand(num_features + 1)  # Retorna pesos aleatorios

# Función para agregar sesgo a X
def agregar_sesgo(X):
    return np.insert(X, 0, 1, axis=1)  # Inserta una columna de unos al principio de X (sesgo)

# Función de activación escalón
def escalon(z):
    return np.where(z >= 0, 1, 0)

# Función principal para entrenar la neurona
def entrenar_neurona(X, Yd, pesos, tasa_aprendizaje, tolerancia_error, iteraciones):
    errores = []  # Lista para almacenar las normas de los errores en cada época
    pesos_evolution = []  # Lista para almacenar la evolución de los pesos en cada época
    epoca=0
    # Ciclo de entrenamiento generacion < self.limiteGeneraciones   
    while epoca < int(iteraciones):
        epoca+=1
        
        # Suma ponderada
        U = np.dot(X, pesos)
        # Función de activación
        Yc = escalon(U)
        # Cálculo del error
        E = Yd - Yc
        # Cálculo de la norma del error
        error_actual = np.linalg.norm(E)
        # Agregar el error actual a la lista
        errores.append(error_actual)
        
        # Si el error es menor o igual a la tolerancia, salir del bucle
        #if error_actual <= tolerancia_error:
        #    break
        
        # Cálculo de los deltas de los pesos
        DeltaW = tasa_aprendizaje * np.dot( E.T,X)
        
        # Actualización de los pesos
        pesos += DeltaW
        
        # Guardar la evolución de los pesos
        pesos_evolution.append(pesos.copy())
        
    return pesos, errores, pesos_evolution

# Función para graficar la norma de los errores en cada época
def graficar_errores(errores):
    # Graficar la norma de los errores en cada época
    plt.plot(range(len(errores)), errores)
    plt.title("Norma de los Errores en Cada Época")
    plt.xlabel("Época")
    plt.ylabel("Norma del Error")
    plt.show()

# Función para graficar la evolución de los pesos a lo largo de todas las épocas
def graficar_pesos_evolution(pesos_evolution):
    pesos_evolution = np.array(pesos_evolution)
    for i in range(pesos_evolution.shape[1]):
        plt.plot(range(pesos_evolution.shape[0]), pesos_evolution[:, i], label=f'Peso {i+1}')
    plt.title("Evolución de los Pesos a lo largo de Todas las Épocas")
    plt.xlabel("Época")
    plt.ylabel("Valor del Peso")
    plt.legend()
    plt.show()

# Método alternativo para cargar datos desde un archivo CSV sin encabezados
def cargar_datos_desde_csv(ruta_csv):
    # Cargar datos desde el archivo CSV
    datos = np.genfromtxt(ruta_csv, delimiter=',')
    # Obtener características de entrada (todas las columnas excepto la última)
    X = datos[:, :-1]
    # Obtener la salida deseada (última columna)
    Yd = datos[:, -1]
    return X, Yd


def ingresarDatos(ruta_csv, tasa_aprendizaje, resultados_texto, iteraciones):
    # Cargar datos desde el archivo CSV
    X, Yd = cargar_datos_desde_csv(ruta_csv)
    # Tolerancia de error
    tolerancia_error = 0

    # Asignar pesos iniciales
    num_features = X.shape[1]
    pesos_iniciales = inicializar_pesos(num_features)
    
    resultados_texto.insert(tk.END, "Configuración de pesos inicial:\n")
    resultados_texto.insert(tk.END, str(pesos_iniciales) + "\n")

    # Agregar sesgo a X
    X_con_sesgo = agregar_sesgo(X)
    pesos_finales, errores, pesos_evolution = entrenar_neurona(X_con_sesgo, Yd, pesos_iniciales, tasa_aprendizaje, tolerancia_error, iteraciones)

#RESULTADOS
    graficar_errores(errores)
    graficar_pesos_evolution(pesos_evolution)
    # Mostrar los pesos finales, tasa de aprendizaje utilizada y el error permisible
    resultados_texto.insert(tk.END, "\nConfiguración de pesos final:\n")
    resultados_texto.insert(tk.END, str(pesos_finales) + "\n")
    resultados_texto.insert(tk.END, "\nTasa de aprendizaje utilizada: " + str(tasa_aprendizaje) + "\n")
    resultados_texto.insert(tk.END, "Error permisible: " + str(tolerancia_error) + "\n")
    resultados_texto.insert(tk.END, "Cantidad de iteraciones: " + str(len(errores)) + "\n")

