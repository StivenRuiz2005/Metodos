import customtkinter as ctk
import numpy as np
from sympy import symbols, diff, lambdify
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NewtonRaphsonSolver(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("Solucionador de Sistemas No Lineales - Newton Raphson")
        self.geometry("1200x800")

        # Variables para las funciones simbólicas
        self.x, self.y = symbols('x y')
        
        # Crear el frame principal con dos columnas
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Frame izquierdo para entradas y resultados
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", padx=10, fill="both", expand=True)

        # Frame derecho para la gráfica
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", padx=10, fill="both", expand=True)

        # Elementos en el frame izquierdo
        self.label_f1 = ctk.CTkLabel(self.left_frame, text="Primera función f1(x,y):")
        self.label_f1.pack(pady=5)
        self.entry_f1 = ctk.CTkEntry(self.left_frame, width=300)
        self.entry_f1.pack(pady=5)
        self.entry_f1.insert(0, "x**2 + y**2 - 4")

        self.label_f2 = ctk.CTkLabel(self.left_frame, text="Segunda función f2(x,y):")
        self.label_f2.pack(pady=5)
        self.entry_f2 = ctk.CTkEntry(self.left_frame, width=300)
        self.entry_f2.pack(pady=5)
        self.entry_f2.insert(0, "x**2 - y - 1")

        # Frame para valores iniciales
        self.init_frame = ctk.CTkFrame(self.left_frame)
        self.init_frame.pack(pady=10)

        self.label_x0 = ctk.CTkLabel(self.init_frame, text="x₀:")
        self.label_x0.pack(side="left", padx=5)
        self.entry_x0 = ctk.CTkEntry(self.init_frame, width=100)
        self.entry_x0.pack(side="left", padx=5)
        self.entry_x0.insert(0, "1")

        self.label_y0 = ctk.CTkLabel(self.init_frame, text="y₀:")
        self.label_y0.pack(side="left", padx=5)
        self.entry_y0 = ctk.CTkEntry(self.init_frame, width=100)
        self.entry_y0.pack(side="left", padx=5)
        self.entry_y0.insert(0, "1")

        self.label_iter = ctk.CTkLabel(self.left_frame, text="Número de iteraciones:")
        self.label_iter.pack(pady=5)
        self.entry_iter = ctk.CTkEntry(self.left_frame, width=100)
        self.entry_iter.pack(pady=5)
        self.entry_iter.insert(0, "10")

        self.solve_button = ctk.CTkButton(self.left_frame, text="Resolver", command=self.solve)
        self.solve_button.pack(pady=10)

        self.result_text = ctk.CTkTextbox(self.left_frame, width=400, height=300)
        self.result_text.pack(pady=10)

        # Configuración del área de la gráfica
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def newton_raphson(self, f1_expr, f2_expr, x0, y0, max_iter):
        f1 = sp.sympify(f1_expr)
        f2 = sp.sympify(f2_expr)

        df1_dx = diff(f1, self.x)
        df1_dy = diff(f1, self.y)
        df2_dx = diff(f2, self.x)
        df2_dy = diff(f2, self.y)

        f1_func = lambdify((self.x, self.y), f1)
        f2_func = lambdify((self.x, self.y), f2)
        df1_dx_func = lambdify((self.x, self.y), df1_dx)
        df1_dy_func = lambdify((self.x, self.y), df1_dy)
        df2_dx_func = lambdify((self.x, self.y), df2_dx)
        df2_dy_func = lambdify((self.x, self.y), df2_dy)

        x_k = float(x0)
        y_k = float(y0)
        results = []

        for k in range(max_iter):
            F = np.array([
                [f1_func(x_k, y_k)],
                [f2_func(x_k, y_k)]
            ])

            J = np.array([
                [df1_dx_func(x_k, y_k), df1_dy_func(x_k, y_k)],
                [df2_dx_func(x_k, y_k), df2_dy_func(x_k, y_k)]
            ])

            try:
                delta = np.linalg.solve(J, -F)
                x_k = x_k + delta[0][0]
                y_k = y_k + delta[1][0]
                results.append((k+1, x_k, y_k, f1_func(x_k, y_k), f2_func(x_k, y_k)))

            except np.linalg.LinAlgError:
                return "Error: Matriz singular encontrada. El método no converge."

        return results

    def plot_results(self, results):
        # Limpiar gráficas anteriores
        self.ax1.clear()
        self.ax2.clear()

        # Extraer datos
        iterations = [r[0] for r in results]
        x_values = [r[1] for r in results]
        y_values = [r[2] for r in results]
        f1_values = [abs(r[3]) for r in results]
        f2_values = [abs(r[4]) for r in results]

        # Gráfica de convergencia de x e y
        self.ax1.plot(iterations, x_values, 'b-o', label='x')
        self.ax1.plot(iterations, y_values, 'r-o', label='y')
        self.ax1.set_title('Convergencia de Variables')
        self.ax1.set_xlabel('Iteración')
        self.ax1.set_ylabel('Valor')
        self.ax1.legend()
        self.ax1.grid(True)

        # Gráfica de error (valores absolutos de f1 y f2)
        self.ax2.semilogy(iterations, f1_values, 'g-o', label='|f₁(x,y)|')
        self.ax2.semilogy(iterations, f2_values, 'm-o', label='|f₂(x,y)|')
        self.ax2.set_title('Error en Funciones')
        self.ax2.set_xlabel('Iteración')
        self.ax2.set_ylabel('Error (escala log)')
        self.ax2.legend()
        self.ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def solve(self):
        try:
            f1_expr = self.entry_f1.get()
            f2_expr = self.entry_f2.get()
            x0 = float(self.entry_x0.get())
            y0 = float(self.entry_y0.get())
            max_iter = int(self.entry_iter.get())

            results = self.newton_raphson(f1_expr, f2_expr, x0, y0, max_iter)

            self.result_text.delete("1.0", "end")

            if isinstance(results, str):
                self.result_text.insert("end", results)
                return

            self.result_text.insert("end", "Iteración | x | y | f1(x,y) | f2(x,y)\n")
            self.result_text.insert("end", "-" * 60 + "\n")
            
            for iter_num, x, y, f1_val, f2_val in results:
                result_line = f"{iter_num:^9} | {x:^10.6f} | {y:^10.6f} | {f1_val:^10.6f} | {f2_val:^10.6f}\n"
                self.result_text.insert("end", result_line)

            # Generar gráficas
            self.plot_results(results)

        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Error: {str(e)}")

if __name__ == "__main__":
    app = NewtonRaphsonSolver()
    app.mainloop()