import customtkinter as ctk
import numpy as np
from sympy import symbols, diff, lambdify
import sympy as sp

class NewtonRaphsonSolver(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("Solucionador de Sistemas No Lineales - Newton Raphson")
        self.geometry("800x600")

        # Variables para las funciones simbólicas
        self.x, self.y = symbols('x y')
        
        # Crear el frame principal
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Entrada para la primera función
        self.label_f1 = ctk.CTkLabel(self.main_frame, text="Primera función f1(x,y):")
        self.label_f1.pack(pady=5)
        self.entry_f1 = ctk.CTkEntry(self.main_frame, width=300)
        self.entry_f1.pack(pady=5)
        self.entry_f1.insert(0, "x**2 + y**2 - 4")  # Ejemplo

        # Entrada para la segunda función
        self.label_f2 = ctk.CTkLabel(self.main_frame, text="Segunda función f2(x,y):")
        self.label_f2.pack(pady=5)
        self.entry_f2 = ctk.CTkEntry(self.main_frame, width=300)
        self.entry_f2.pack(pady=5)
        self.entry_f2.insert(0, "x**2 - y - 1")  # Ejemplo

        # Frame para valores iniciales
        self.init_frame = ctk.CTkFrame(self.main_frame)
        self.init_frame.pack(pady=10)

        # Valor inicial de x
        self.label_x0 = ctk.CTkLabel(self.init_frame, text="x₀:")
        self.label_x0.pack(side="left", padx=5)
        self.entry_x0 = ctk.CTkEntry(self.init_frame, width=100)
        self.entry_x0.pack(side="left", padx=5)
        self.entry_x0.insert(0, "1")

        # Valor inicial de y
        self.label_y0 = ctk.CTkLabel(self.init_frame, text="y₀:")
        self.label_y0.pack(side="left", padx=5)
        self.entry_y0 = ctk.CTkEntry(self.init_frame, width=100)
        self.entry_y0.pack(side="left", padx=5)
        self.entry_y0.insert(0, "1")

        # Número de iteraciones
        self.label_iter = ctk.CTkLabel(self.main_frame, text="Número de iteraciones:")
        self.label_iter.pack(pady=5)
        self.entry_iter = ctk.CTkEntry(self.main_frame, width=100)
        self.entry_iter.pack(pady=5)
        self.entry_iter.insert(0, "10")

        # Botón para resolver
        self.solve_button = ctk.CTkButton(self.main_frame, text="Resolver", command=self.solve)
        self.solve_button.pack(pady=10)

        # Área de resultados
        self.result_text = ctk.CTkTextbox(self.main_frame, width=700, height=300)
        self.result_text.pack(pady=10)

    def newton_raphson(self, f1_expr, f2_expr, x0, y0, max_iter):
        # Crear funciones simbólicas
        f1 = sp.sympify(f1_expr)
        f2 = sp.sympify(f2_expr)

        # Calcular las derivadas parciales
        df1_dx = diff(f1, self.x)
        df1_dy = diff(f1, self.y)
        df2_dx = diff(f2, self.x)
        df2_dy = diff(f2, self.y)

        # Convertir expresiones simbólicas a funciones numéricas
        f1_func = lambdify((self.x, self.y), f1)
        f2_func = lambdify((self.x, self.y), f2)
        df1_dx_func = lambdify((self.x, self.y), df1_dx)
        df1_dy_func = lambdify((self.x, self.y), df1_dy)
        df2_dx_func = lambdify((self.x, self.y), df2_dx)
        df2_dy_func = lambdify((self.x, self.y), df2_dy)

        # Inicializar variables
        x_k = float(x0)
        y_k = float(y0)
        results = []

        for k in range(max_iter):
            # Evaluar funciones en el punto actual
            F = np.array([
                [f1_func(x_k, y_k)],
                [f2_func(x_k, y_k)]
            ])

            # Evaluar matriz jacobiana
            J = np.array([
                [df1_dx_func(x_k, y_k), df1_dy_func(x_k, y_k)],
                [df2_dx_func(x_k, y_k), df2_dy_func(x_k, y_k)]
            ])

            try:
                # Resolver el sistema para obtener el incremento
                delta = np.linalg.solve(J, -F)
                
                # Actualizar valores
                x_k = x_k + delta[0][0]
                y_k = y_k + delta[1][0]
                
                # Guardar resultados
                results.append((k+1, x_k, y_k, f1_func(x_k, y_k), f2_func(x_k, y_k)))

            except np.linalg.LinAlgError:
                return "Error: Matriz singular encontrada. El método no converge."

        return results

    def solve(self):
        try:
            # Obtener valores de entrada
            f1_expr = self.entry_f1.get()
            f2_expr = self.entry_f2.get()
            x0 = float(self.entry_x0.get())
            y0 = float(self.entry_y0.get())
            max_iter = int(self.entry_iter.get())

            # Resolver el sistema
            results = self.newton_raphson(f1_expr, f2_expr, x0, y0, max_iter)

            # Limpiar área de resultados
            self.result_text.delete("1.0", "end")

            if isinstance(results, str):
                self.result_text.insert("end", results)
                return

            # Mostrar resultados
            self.result_text.insert("end", "Iteración | x | y | f1(x,y) | f2(x,y)\n")
            self.result_text.insert("end", "-" * 60 + "\n")
            
            for iter_num, x, y, f1_val, f2_val in results:
                result_line = f"{iter_num:^9} | {x:^10.6f} | {y:^10.6f} | {f1_val:^10.6f} | {f2_val:^10.6f}\n"
                self.result_text.insert("end", result_line)

        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {str(e)}")

if __name__ == "__main__":
    app = NewtonRaphsonSolver()
    app.mainloop()