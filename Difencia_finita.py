import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import customtkinter as ctk
from tabulate import tabulate

# Configuración inicial de CustomTkinter
ctk.set_appearance_mode("System")  # "Dark" o "Light"
ctk.set_default_color_theme("blue")  # Tema de color

# Función principal para resolver la ecuación diferencial con diferencias finitas
def resolver_diferencias_finitas():
    try:
        # Obtener entradas del usuario
        ecuacion_str = entrada_ecuacion.get()  # Ecuación diferencial ingresada como string
        L = float(entrada_longitud.get())  # Longitud del dominio
        n = int(entrada_n.get())  # Número de puntos discretizados
        y0 = float(entrada_y0.get())  # Condición de frontera y(0)
        yn = float(entrada_yn.get())  # Condición de frontera y(L)
        
        # Crear el símbolo x
        x = sp.Symbol('x')
        
        # Convertir la ecuación ingresada a una expresión simbólica
        ecuacion = sp.sympify(ecuacion_str)
        
        # Discretización
        h = L / (n - 1)  # Paso de discretización
        
        # Matriz A (coeficientes)
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Llenar la matriz A y el vector b según las diferencias finitas
        for i in range(1, n-1):
            A[i, i-1] = 4
            A[i, i] = -2
            A[i, i+1] = 1
            b[i] = -x.subs(x, i * h)  # Evaluar la ecuación en el punto correspondiente (x = i * h)
        
        # Condiciones de frontera
        A[0, 0] = 1
        A[-1, -1] = 1
        b[0] = y0
        b[-1] = yn
        
        # Resolver el sistema de ecuaciones
        y = np.linalg.solve(A, b)
        
        # Mostrar resultados en formato tabla
        resultados = [
            [i, y[i]]
            for i in range(n)
        ]
        print(tabulate(resultados, headers=["Punto", "Valor de y"]))
        
        # Graficar la solución
        fig, ax = plt.subplots(figsize=(8, 5))
        x_vals = np.linspace(0, L, n)
        ax.plot(x_vals, y, label="Solución de la Ecuación Diferencial", color="blue")
        ax.set_title("Solución por Diferencias Finitas")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en la interfaz
        for widget in frame_grafica.winfo_children():
            widget.destroy()  # Limpiar gráfico previo
        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)

    except Exception as e:
        etiqueta_resultado.configure(text=f"Error: {e}")

# Configuración de la ventana principal
ventana = ctk.CTk()
ventana.title("Método de Diferencias Finitas para Ecuaciones Diferenciales")
ventana.geometry("900x600")

# Frame de entradas
frame_entradas = ctk.CTkFrame(ventana)
frame_entradas.pack(side=ctk.LEFT, fill=ctk.BOTH, padx=10, pady=10)

# Entradas de usuario
ctk.CTkLabel(frame_entradas, text="Ecuación Diferencial (en términos de y' y y''):").pack(pady=5)
entrada_ecuacion = ctk.CTkEntry(frame_entradas, width=200)
entrada_ecuacion.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Longitud del dominio (L):").pack(pady=5)
entrada_longitud = ctk.CTkEntry(frame_entradas, width=200)
entrada_longitud.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Número de puntos n:").pack(pady=5)
entrada_n = ctk.CTkEntry(frame_entradas, width=200)
entrada_n.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Valor de y(0):").pack(pady=5)
entrada_y0 = ctk.CTkEntry(frame_entradas, width=200)
entrada_y0.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Valor de y(L):").pack(pady=5)
entrada_yn = ctk.CTkEntry(frame_entradas, width=200)
entrada_yn.pack(pady=5)

# Botón para calcular
boton_calcular = ctk.CTkButton(frame_entradas, text="Calcular", command=resolver_diferencias_finitas)
boton_calcular.pack(pady=20)

# Etiqueta para errores o resultados
etiqueta_resultado = ctk.CTkLabel(frame_entradas, text="")
etiqueta_resultado.pack(pady=5)

# Frame para la gráfica
frame_grafica = ctk.CTkFrame(ventana)
frame_grafica.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=10, pady=10)

# Iniciar la ventana
ventana.mainloop()
