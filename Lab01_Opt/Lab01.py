import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sympy as spy
import time

# Función de beneficio: Z = 10x + 15y
def profit_function(x, y):
    return 10 * x + 15 * y

# Interfaz Streamlit
st.title("Optimización de la Producción de la Fábrica")

# Agregar el texto después del título
st.write(
    """
    Una fábrica produce dos tipos de productos, P1 y P2. Cada unidad de P1 requiere 2 horas de
    trabajo y 1 kg de material, mientras que cada unidad de P2 requiere 1 hora de trabajo y 2 kg de
    material. La fábrica tiene disponibles 40 horas de trabajo y 30 kg de material al día. Si el beneficio
    por unidad de P1 es de 10 y por unidad de P2 es de 15, ¿cuántas unidades de cada producto deben
    producirse para maximizar el beneficio?
    """
)

# Entrada para las unidades de P1 y P2
x_input = st.number_input("Ingrese x (unidades de P1)", value=0.0)
y_input = st.number_input("Ingrese y (unidades de P2)", value=0.0)

# Entrada para las restricciones
st.write("### Restricciones")
work_hours = st.number_input("Horas de trabajo disponibles", value=40.0)
material_kg = st.number_input("Kg de material disponibles", value=30.0)

# Restricciones
def is_feasible(x, y, work_hours, material_kg):
    return (2 * x + y <= work_hours) and (x + 2 * y <= material_kg) and (x >= 0) and (y >= 0)

# Cálculo del valor de la función de beneficio
if is_feasible(x_input, y_input, work_hours, material_kg):
    profit = profit_function(x_input, y_input)
    st.write(f"Valor de la función de beneficio: {profit}")
else:
    st.write("El punto no cumple con las restricciones")

# Generación de la región factible
x = np.linspace(0, 20, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)
Z = profit_function(X, Y)

# Restricciones
constraint1 = 2 * X + Y <= work_hours
constraint2 = X + 2 * Y <= material_kg
feasible_mask = constraint1 & constraint2 & (X >= 0) & (Y >= 0)

# Graficar
fig, ax = plt.subplots()
# ax.contourf(X, Y, Z, levels=50, cmap='viridis')  # Eliminar o comentar esta línea
ax.plot(x, (work_hours - 2 * x), label=f"2x + y ≤ {work_hours} (horas de trabajo)")
ax.plot(x, (material_kg - x) / 2, label=f"x + 2y ≤ {material_kg} (material)")
ax.fill_between(x, 0, np.minimum((work_hours - 2 * x), (material_kg - x) / 2), where=(x >= 0), color='lightgreen', alpha=0.5, label="Región factible")
plt.plot(x_input, y_input, 'ro', label=f"Punto ({x_input}, {y_input})")
plt.xlabel("Unidades de P1")
plt.ylabel("Unidades de P2")
plt.legend()
st.pyplot(fig)

# Generar una matriz dispersa aleatoria
def generate_sparse_matrix(size, density=0.1):
    matrix = np.random.choice([0, 1], size=(size, size), p=[1 - density, density]) * np.random.randint(1, 10, size=(size, size))
    return matrix

# Medir el tiempo de ejecución
def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Interfaz Streamlit
st.title("Operaciones con Matrices")

# Ajustar el tamaño máximo de la matriz a 5000
size = st.slider("Tamaño de la matriz", min_value=10, max_value=5000, value=100)
density = st.slider("Densidad", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Generar la matriz
matrix = generate_sparse_matrix(size, density)

# Selección de la operación
operation = st.selectbox("Seleccione la operación:", ["Suma", "Multiplicación"])

# Selección del método de representación
method = st.selectbox("Método de representación:", ["Matriz Densa", "CSR Manual", "SciPy CSR", "SciPy COO", "SciPy CSC"])

# Convertir la matriz a la representación seleccionada
if method == "Matriz Densa":
    matrix_a = matrix
    matrix_b = matrix  # Usamos la misma matriz para la operación
elif method == "CSR Manual":
    class SparseMatrixCSR:
        def __init__(self, matrix):
            self.values = []
            self.col_indices = []
            self.row_ptr = [0]

            for row in matrix:
                non_zero_count = 0
                for j, value in enumerate(row):
                    if value != 0:
                        self.values.append(value)
                        self.col_indices.append(j)
                        non_zero_count += 1
                self.row_ptr.append(self.row_ptr[-1] + non_zero_count)

        def to_dense(self, shape):
            dense_matrix = np.zeros(shape)
            for i in range(len(self.row_ptr) - 1):
                start = self.row_ptr[i]
                end = self.row_ptr[i + 1]
                for j in range(start, end):
                    col_index = self.col_indices[j]
                    dense_matrix[i][col_index] = self.values[j]
            return dense_matrix

    sparse_manual = SparseMatrixCSR(matrix)
    matrix_a = sparse_manual
    matrix_b = sparse_manual  # Usamos la misma matriz para la operación
elif method == "SciPy CSR":
    matrix_a = sp.csr_matrix(matrix)
    matrix_b = sp.csr_matrix(matrix)
elif method == "SciPy COO":
    matrix_a = sp.coo_matrix(matrix)
    matrix_b = sp.coo_matrix(matrix)
elif method == "SciPy CSC":
    matrix_a = sp.csc_matrix(matrix)
    matrix_b = sp.csc_matrix(matrix)

# Realizar la operación seleccionada
if st.button("Realizar operación"):
    if operation == "Suma":
        if method == "Matriz Densa":
            result, time_op = measure_time(np.add, matrix_a, matrix_b)
        elif method == "CSR Manual":
            dense_a = matrix_a.to_dense(matrix.shape)
            dense_b = matrix_b.to_dense(matrix.shape)
            result, time_op = measure_time(np.add, dense_a, dense_b)
        elif method == "SciPy CSR":
            result, time_op = measure_time(sp.csr_matrix.__add__, matrix_a, matrix_b)
        elif method == "SciPy COO":
            result, time_op = measure_time(sp.coo_matrix.__add__, matrix_a, matrix_b)
        elif method == "SciPy CSC":
            result, time_op = measure_time(sp.csc_matrix.__add__, matrix_a, matrix_b)
    elif operation == "Multiplicación":
        if method == "Matriz Densa":
            result, time_op = measure_time(np.dot, matrix_a, matrix_b)
        elif method == "CSR Manual":
            dense_a = matrix_a.to_dense(matrix.shape)
            dense_b = matrix_b.to_dense(matrix.shape)
            result, time_op = measure_time(np.dot, dense_a, dense_b)
        elif method == "SciPy CSR":
            result, time_op = measure_time(sp.csr_matrix.dot, matrix_a, matrix_b)
        elif method == "SciPy COO":
            result, time_op = measure_time(sp.coo_matrix.dot, matrix_a, matrix_b)
        elif method == "SciPy CSC":
            result, time_op = measure_time(sp.csc_matrix.dot, matrix_a, matrix_b)

    st.write(f"Tiempo de operación ({operation} con {method}): {time_op:.6f} segundos")




def taylor_series(func, x0, n, x):
    """Calcula la serie de Taylor de la función func en torno a x0 con n términos."""
    series = func.series(x, x0, n).removeO()
    return series

def plot_taylor_approximation(func, x0, n, x_range=(-5, 5)):
    """Grafica la función original y su aproximación con series de Taylor."""
    x = spy.Symbol('x')
    taylor_approx = taylor_series(func, x0, n, x)
    
    # Convertimos la expresión simbólica en función numérica
    f_lambdified = spy.lambdify(x, func, 'numpy')
    taylor_lambdified = spy.lambdify(x, taylor_approx, 'numpy')
    
    # Rango de valores para graficar
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f_lambdified(x_vals)
    y_taylor_vals = taylor_lambdified(x_vals)
    
    # Graficamos
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, label='Función original', linewidth=2)
    ax.plot(x_vals, y_taylor_vals, label=f'Aproximación de Taylor (n={n})', linestyle='dashed')
    ax.axvline(x=x0, color='gray', linestyle='dotted', label=f'Expansión en x={x0}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Serie de Taylor en x0={x0} con {n} términos')
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)

# Streamlit UI
st.title("Expansión en Series de Taylor")

functions = {
    'e^x': spy.exp(spy.Symbol('x')),
    'sin(x)': spy.sin(spy.Symbol('x')),
    'cos(x)': spy.cos(spy.Symbol('x')),
    'ln(1+x)': spy.ln(1 + spy.Symbol('x')),
    '1/(1+x)': (1 + spy.Symbol('x'))**(-1)
}

func_choice = st.selectbox("Seleccione una función:", list(functions.keys()))
x0 = st.number_input("Ingrese el punto de expansión x0:", value=0.0, step=0.1)
n = st.slider("Seleccione el número de términos de la serie de Taylor:", min_value=1, max_value=20, value=5)

if st.button("Generar Gráfica"):
    # Validación para evitar valores problemáticos
    if func_choice == "ln(1+x)" and x0 <= -1:
        st.error("El punto de expansión x0 debe ser mayor que -1 para ln(1+x).")
    elif func_choice == "1/(1+x)" and x0 == -1:
        st.error("El punto de expansión x0 no puede ser -1 para 1/(1+x) porque causa una división por cero.")
    else:
        plot_taylor_approximation(functions[func_choice], x0, n)

#A partir de aca se resuelve el punto 4

# Función a minimizar
def f(x):
    return (x - 2) ** 2 + 1

# Derivadas para Newton y Gradiente Descendente
def df(x):
    return 2 * (x - 2)

def d2f(x):
    return 2  # Segunda derivada constante

# 🔹 Método de Newton
def newton_method(x0, tol=1e-5, max_iter=50):
    x = x0
    history = [x]
    
    for _ in range(max_iter):
        x_new = x - df(x) / d2f(x)
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            break
        x = x_new

    return x_new, len(history), history

# Gradiente Descendente
def gradient_descent(x0, alpha=0.1, tol=1e-5, max_iter=100):
    x = x0
    history = [x]
    
    for _ in range(max_iter):
        x_new = x - alpha * df(x)
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            break
        x = x_new

    return x_new, len(history), history

# Búsqueda Unidireccional
def unidirectional_search(x0, step=0.1, tol=1e-5):
    x = x0
    history = [x]
    
    direction = 1 if f(x + step) < f(x) else -1  # Dirección inicial
    
    while True:
        x_new = x + direction * step
        history.append(x_new)

        if f(x_new) >= f(x):  # Si ya no mejora, se detiene
            break
        x = x_new

    return x, len(history), history

#  Interfaz Streamlit
st.title("Comparación de Métodos de Optimización")

# Muestra la función utilizada
st.write("### Función utilizada:")
st.latex(r"f(x) = (x - 2)^2 + 1")

# Selección del método
method = st.selectbox("Selecciona un método:", ["Newton", "Gradiente Descendente", "Búsqueda Unidireccional"])


# Parámetros generales
x0 = st.number_input("Punto inicial (x0):", value=-5.0, step=0.1)

# Parámetros específicos por método
if method == "Gradiente Descendente":
    alpha = st.slider("Tasa de aprendizaje (alpha):", 0.001, 1.0, 0.1)
    result = gradient_descent(x0, alpha=alpha)
elif method == "Búsqueda Unidireccional":
    step = st.slider("Tamaño del paso:", 0.01, 1.0, 0.1)
    result = unidirectional_search(x0, step=step)
else:
    result = newton_method(x0)

# Desplegar resultados
x_min, iters, history = result
st.write(f"🔹 **Mínimo encontrado en:** {x_min:.5f}")
st.write(f"🔹 **Iteraciones realizadas:** {iters}")

# Gráfico
x_vals = np.linspace(-6, 6, 100)
y_vals = f(x_vals)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_vals, y_vals, 'k--', alpha=0.5, label="Función f(x)")
ax.plot(history, f(np.array(history)), marker='o', linestyle='-', label=method)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
ax.set_title("Proceso de optimización")

st.pyplot(fig)
