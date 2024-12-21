import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import ttk

# Configuración inicial de CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

def crear_campos_coeficientes(*args):
    # Limpiar campos anteriores
    for widget in frame_coeficientes.winfo_children():
        widget.destroy()
    
    try:
        grado = int(entrada_grado.get())
        grado_x = int(entrada_grado_x.get())
        if grado < 1 or grado > 4:
            etiqueta_resultado.configure(text="El grado de y debe estar entre 1 y 4")
            return
        if grado_x < 0 or grado_x > 4:
            etiqueta_resultado.configure(text="El grado de x debe estar entre 0 y 4")
            return
            
        # Frame para términos con y
        frame_terminos_y = ctk.CTkFrame(frame_coeficientes)
        frame_terminos_y.pack(side=ctk.LEFT, padx=10, pady=5)
        ctk.CTkLabel(frame_terminos_y, text="Términos con y:").pack(pady=5)
        
        # Crear campos para cada derivada de y
        entradas_coef_y = []
        for i in range(grado, -1, -1):
            if i == 0:
                label = ctk.CTkLabel(frame_terminos_y, text=f"Coeficiente de y:")
            else:
                label = ctk.CTkLabel(frame_terminos_y, text=f"Coeficiente de y^({i}):")
            label.pack(pady=2)
            entrada = ctk.CTkEntry(frame_terminos_y, width=100)
            entrada.pack(pady=2)
            entrada.insert(0, "0")
            entradas_coef_y.append(entrada)
            
        # Frame para términos con x
        frame_terminos_x = ctk.CTkFrame(frame_coeficientes)
        frame_terminos_x.pack(side=ctk.LEFT, padx=10, pady=5)
        ctk.CTkLabel(frame_terminos_x, text="Términos con x:").pack(pady=5)
        
        # Crear campos para cada potencia de x
        entradas_coef_x = []
        for i in range(grado_x, -1, -1):
            if i == 0:
                label = ctk.CTkLabel(frame_terminos_x, text=f"Término independiente:")
            else:
                label = ctk.CTkLabel(frame_terminos_x, text=f"Coeficiente de x^{i}:")
            label.pack(pady=2)
            entrada = ctk.CTkEntry(frame_terminos_x, width=100)
            entrada.pack(pady=2)
            entrada.insert(0, "0")
            entradas_coef_x.append(entrada)
        
        # Guardar las entradas en variables globales
        global coef_entries_y, coef_entries_x
        coef_entries_y = entradas_coef_y
        coef_entries_x = entradas_coef_x
        
    except ValueError:
        etiqueta_resultado.configure(text="Por favor, ingrese números válidos para los grados")

def construir_ecuacion():
    try:
        # Obtener el grado y crear los símbolos
        grado = int(entrada_grado.get())
        grado_x = int(entrada_grado_x.get())
        x = sp.Symbol('x')
        y = sp.Function('y')(x)
        
        # Construir la ecuación - términos con y
        ecuacion = 0
        for i, entrada in enumerate(coef_entries_y):
            coef = float(entrada.get())
            if coef != 0:
                potencia = grado - i
                if potencia == 0:
                    ecuacion += coef * y
                else:
                    ecuacion += coef * y.diff(x, potencia)
                    
        # Agregar términos con x
        for i, entrada in enumerate(coef_entries_x):
            coef = float(entrada.get())
            if coef != 0:
                potencia = grado_x - i
                ecuacion += coef * x**potencia
            
        return ecuacion
    except Exception as e:
        etiqueta_resultado.configure(text=f"Error al construir la ecuación: {str(e)}")
        return None

def resolver_diferencias_finitas():
    try:
        # Obtener valores de las condiciones de frontera y dominio
        L = float(entrada_longitud.get())
        n = int(entrada_n.get())
        y0 = float(entrada_y0.get())
        yn = float(entrada_yn.get())

        # Construir la ecuación
        ecuacion = construir_ecuacion()
        if ecuacion is None:
            return
            
        x = sp.Symbol('x')
        y = sp.Function('y')(x)
        
        # Convertir a ecuación estándar
        eq = sp.Eq(ecuacion, 0)
        eq_expanded = sp.expand(eq.lhs)
        
        # Obtener coeficientes
        coef_y2 = eq_expanded.coeff(y.diff(x, 2))
        if coef_y2 == 0:
            etiqueta_resultado.configure(text="Error: El coeficiente de y'' debe ser distinto de cero")
            return
            
        coef_y1 = eq_expanded.coeff(y.diff(x))
        coef_y = eq_expanded.coeff(y)
        
        # Función para obtener términos independientes
        def get_independent_terms(expr, y):
            return expr.subs({y.diff(x, 2): 0, y.diff(x): 0, y: 0})
        
        f_term = -get_independent_terms(eq_expanded, y)
        
        # Normalizar la ecuación
        p_x = sp.lambdify(x, coef_y1/coef_y2, "numpy")
        q_x = sp.lambdify(x, coef_y/coef_y2, "numpy")
        f_x = sp.lambdify(x, f_term/coef_y2, "numpy")

        # Discretización
        puntos_totales = n + 2
        h = L / (puntos_totales - 1)

        # Crear la matriz A y el vector b
        A = np.zeros((n, n))
        b = np.zeros(n)
        x_vals = np.linspace(0, L, puntos_totales)

        for i in range(n):
            xi = x_vals[i + 1]

            try:
                p_val = p_x(xi)
                q_val = q_x(xi)
                f_val = f_x(xi)
            except Exception as e:
                etiqueta_resultado.configure(text=f"Error al evaluar coeficientes en x={xi}: {str(e)}")
                return

            A[i, i] = -2 + h**2 * q_val
            if i > 0:
                A[i, i - 1] = 1 - (h / 2) * p_val
            if i < n - 1:
                A[i, i + 1] = 1 + (h / 2) * p_val

            b[i] = h**2 * f_val

        b[0] -= (1 - (h / 2) * p_x(x_vals[1])) * y0
        b[-1] -= (1 + (h / 2) * p_x(x_vals[-2])) * yn

        # Resolver el sistema
        try:
            y_interior = np.linalg.solve(A, b)
            y_solucion = np.concatenate(([y0], y_interior, [yn]))
            
            # Limpiar tabla previa
            for row in tabla.get_children():
                tabla.delete(row)

            # Agregar resultados a la tabla
            for i in range(puntos_totales):
                tabla.insert("", "end", values=(f"{x_vals[i]:.10f}", f"{y_solucion[i]:.10f}"))

            # Graficar
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_vals, y_solucion, 'bo-', label='Solución numérica')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.legend()

            # Mostrar gráfica
            for widget in frame_grafica.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
            
            etiqueta_resultado.configure(text="Solución completada exitosamente")
            
        except Exception as e:
            etiqueta_resultado.configure(text=f"Error al resolver el sistema: {str(e)}")

    except Exception as e:
        etiqueta_resultado.configure(text=f"Error general: {str(e)}")

# Configuración de la ventana principal
ventana = ctk.CTk()
ventana.title("Método de Diferencias Finitas")
ventana.geometry("1200x800")
ventana.state("zoomed")

# Frame principal izquierdo
frame_izquierdo = ctk.CTkFrame(ventana)
frame_izquierdo.pack(side=ctk.LEFT, fill=ctk.Y, padx=10, pady=10)

# Frame para grados de la ecuación
frame_grado = ctk.CTkFrame(frame_izquierdo)
frame_grado.pack(fill=ctk.X, padx=10, pady=5)

# Grado de y
ctk.CTkLabel(frame_grado, text="Grado más alto de y (1-4):").pack(pady=2)
entrada_grado = ctk.CTkEntry(frame_grado, width=50)
entrada_grado.pack(pady=2)

# Grado de x
ctk.CTkLabel(frame_grado, text="Grado más alto de x (0-4):").pack(pady=2)
entrada_grado_x = ctk.CTkEntry(frame_grado, width=50)
entrada_grado_x.pack(pady=2)

boton_crear_campos = ctk.CTkButton(frame_grado, text="Crear campos", 
                                  command=lambda: crear_campos_coeficientes())
boton_crear_campos.pack(pady=5)

# Frame para coeficientes
frame_coeficientes = ctk.CTkFrame(frame_izquierdo)
frame_coeficientes.pack(fill=ctk.X, padx=10, pady=5)

# Frame para condiciones de frontera
frame_condiciones = ctk.CTkFrame(frame_izquierdo)
frame_condiciones.pack(fill=ctk.X, padx=10, pady=5)

ctk.CTkLabel(frame_condiciones, text="Longitud del dominio L:").pack(pady=2)
entrada_longitud = ctk.CTkEntry(frame_condiciones, width=100)
entrada_longitud.pack(pady=2)

ctk.CTkLabel(frame_condiciones, text="Número de puntos interiores (n):").pack(pady=2)
entrada_n = ctk.CTkEntry(frame_condiciones, width=100)
entrada_n.pack(pady=2)

ctk.CTkLabel(frame_condiciones, text="Condición y(0):").pack(pady=2)
entrada_y0 = ctk.CTkEntry(frame_condiciones, width=100)
entrada_y0.pack(pady=2)

ctk.CTkLabel(frame_condiciones, text="Condición y(L):").pack(pady=2)
entrada_yn = ctk.CTkEntry(frame_condiciones, width=100)
entrada_yn.pack(pady=2)

# Botón para calcular
boton_calcular = ctk.CTkButton(frame_izquierdo, text="Resolver", command=resolver_diferencias_finitas)
boton_calcular.pack(pady=20)

# Etiqueta para resultados
etiqueta_resultado = ctk.CTkLabel(frame_izquierdo, text="", wraplength=350)
etiqueta_resultado.pack(pady=5)

# Frame para resultados (derecha)
frame_resultados = ctk.CTkFrame(ventana)
frame_resultados.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=10, pady=10)

# Frame para la gráfica
frame_grafica = ctk.CTkFrame(frame_resultados)
frame_grafica.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

# Tabla de resultados
tabla = ttk.Treeview(frame_resultados, columns=("x", "y"), show="headings", height=10)
tabla.heading("x", text="x")
tabla.heading("y", text="y")
tabla.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

# Variables globales para las entradas de coeficientes
coef_entries_y = []
coef_entries_x = []

# Iniciar la ventana
ventana.mainloop()
