import tkinter as tk
from tkinter import filedialog
import os
import Binario

class VentanaInterfaz:
    def __init__(self, root):
        self.root = root
        self.root.title("Interfaz para Perceptrón")
        self.root.geometry("600x350")

        self.label_archivo = tk.Label(root, text="Ruta del archivo CSV:")
        self.label_archivo.pack()

        self.entry_ruta_archivo = tk.Entry(root, width=50)
        self.entry_ruta_archivo.pack()

        self.btn_seleccionar = tk.Button(root, text="Seleccionar archivo", command=self.seleccionar_archivo)
        self.btn_seleccionar.pack()

        self.label_tasa_aprendizaje = tk.Label(root, text="Tasa de aprendizaje:")
        self.label_tasa_aprendizaje.pack()
        self.entry_tasa_aprendizaje = tk.Entry(root)
        self.entry_tasa_aprendizaje.pack()

        self.label_Iteraciones = tk.Label(root, text="Iteraciones:")
        self.label_Iteraciones.pack()
        self.entry_Iteraciones = tk.Entry(root)
        self.entry_Iteraciones.pack()

        self.btn_entrenar = tk.Button(root, text="Entrenar", command=self.entrenar_perceptron)
        self.btn_entrenar.pack()

        self.resultados_texto = tk.Text(root, height=10, width=50)
        self.resultados_texto.pack()

    def seleccionar_archivo(self):
        ruta_archivo = filedialog.askopenfilename(initialdir=os.getcwd(), title="Seleccionar archivo", filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
        if ruta_archivo:
            self.entry_ruta_archivo.delete(0, tk.END)
            self.entry_ruta_archivo.insert(tk.END, ruta_archivo)

    def entrenar_perceptron(self):
        ruta_archivo = self.entry_ruta_archivo.get()
        tasa_aprendizaje = self.entry_tasa_aprendizaje.get()
        iteraciones = self.entry_Iteraciones.get()
        try:
            Binario.ingresarDatos(ruta_archivo, float(tasa_aprendizaje), self.resultados_texto,iteraciones)
        except Exception as e:
            self.mostrar_resultado(f"Error al entrenar el perceptrón: {str(e)}")

    def mostrar_resultado(self, mensaje):
        self.resultados_texto.insert(tk.END, mensaje + "\n")
        self.resultados_texto.see(tk.END)

def main():
    root = tk.Tk()
    ventana = VentanaInterfaz(root)
    root.mainloop()

if __name__ == "__main__":
    main()
