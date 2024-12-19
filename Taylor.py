import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import customtkinter as ctk
from tabulate import tabulate  # Para mostrar los resultados en formato tabla

# Configuración inicial de CustomTkinter
ctk.set_appearance_mode("System")  # "Dark" o "Light"
ctk.set_default_color_theme("blue")  # Tema de color

# Función principal para calcular la Serie de Taylor
def calcular_taylor():
    try:
        # Obtener entradas del usuario
        func_str = entrada_funcion.get()
        a = float(entrada_a.get())
        x_val = float(entrada_x.get())
        n = int(entrada_n.get())
        
        # Definir la función simbólica
        x = sp.Symbol('x')
        func = sp.sympify(func_str)
        
        # Construir la Serie de Taylor y calcular valores para cada iteración
        taylor_series = 0
        valores_aproximados = []
        errores_reales = []
        errores_relativos = []

        # Valor exacto de la función en x_val
        valor_exacto = sp.lambdify(x, func, 'numpy')(x_val)

        for k in range(1, n + 1):
            deriv = sp.diff(func, x, k - 1)  # Derivada (k-1)-ésima
            term = (deriv.subs(x, a) / sp.factorial(k - 1)) * (x - a)**(k - 1)
            taylor_series += term

            # Evaluar la serie en x_val
            taylor_numeric = sp.lambdify(x, taylor_series, 'numpy')
            valor_aprox = taylor_numeric(x_val)

            # Calcular errores
            error_real = abs(valor_exacto - valor_aprox)
            error_relativo = (error_real / abs(valor_exacto)) * 100 if valor_exacto != 0 else 0

            # Guardar resultados
            valores_aproximados.append(valor_aprox)
            errores_reales.append(error_real)
            errores_relativos.append(error_relativo)

        # Mostrar resultados en la tabla en la interfaz gráfica con 10 decimales
        tabla_resultados = [
            f"{k} | {valores_aproximados[k - 1]:.10f} | {errores_reales[k - 1]:.10f} | {errores_relativos[k - 1]:.10f}%"
            for k in range(1, n + 1)
        ]
        # Usando configure en lugar de config
        etiqueta_resultado.configure(text="\n".join(tabla_resultados))

        # Graficar los errores y aproximaciones
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, n + 1), valores_aproximados, label="Aproximación", color="blue", marker="o")
        ax.axhline(valor_exacto, color="red", linestyle="--", label="Valor Exacto")
        ax.set_title("Convergencia de la Serie de Taylor")
        ax.set_xlabel("Número de términos (n)")
        ax.set_ylabel("Valor Aproximado")
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
ventana.title("Aproximación de Series de Taylor")
ventana.geometry("1000x600")  # Ajustado para dar espacio para la tabla

# Frame de entradas
frame_entradas = ctk.CTkFrame(ventana)
frame_entradas.pack(side=ctk.LEFT, fill=ctk.BOTH, padx=10, pady=10)

# Entradas de usuario
ctk.CTkLabel(frame_entradas, text="Función f(x):").pack(pady=5)
entrada_funcion = ctk.CTkEntry(frame_entradas, width=200)
entrada_funcion.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Punto a (centro de la serie):").pack(pady=5)
entrada_a = ctk.CTkEntry(frame_entradas, width=200)
entrada_a.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Punto x (para evaluar):").pack(pady=5)
entrada_x = ctk.CTkEntry(frame_entradas, width=200)
entrada_x.pack(pady=5)

ctk.CTkLabel(frame_entradas, text="Número máximo de términos n:").pack(pady=5)
entrada_n = ctk.CTkEntry(frame_entradas, width=200)
entrada_n.pack(pady=5)

# Botón para calcular
boton_calcular = ctk.CTkButton(frame_entradas, text="Calcular", command=calcular_taylor)
boton_calcular.pack(pady=20)

# Etiqueta para errores o resultados
etiqueta_resultado = ctk.CTkLabel(frame_entradas, text="", width=300, anchor="w")
etiqueta_resultado.pack(pady=5)

# Frame para la gráfica
frame_grafica = ctk.CTkFrame(ventana)
frame_grafica.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=10, pady=10)

# Frame para la tabla
frame_tabla = ctk.CTkFrame(ventana)
frame_tabla.pack(side=ctk.RIGHT, fill=ctk.BOTH, padx=10, pady=10, expand=True)

# Etiqueta para mostrar la tabla
etiqueta_resultado = ctk.CTkLabel(frame_tabla, text="", width=300, anchor="w", justify="left")
etiqueta_resultado.pack(padx=10, pady=10)

# Iniciar la ventana
ventana.mainloop()
