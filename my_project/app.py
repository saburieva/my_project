from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
from sympy.abc import x, y
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)


def analyze_critical_points(f_expression, g_expression):
    """Анализирует особые точки системы дифференциальных уравнений"""
    try:
        # Парсим выражения
        f = sp.sympify(f_expression)
        g = sp.sympify(g_expression)

        # Находим особые точки (где f=0 и g=0)
        critical_points = sp.solve([f, g], (x, y), dict=True)

        results = []

        for point in critical_points:
            x0 = point[x]
            y0 = point[y]

            # Вычисляем матрицу Якоби
            J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)],
                           [sp.diff(g, x), sp.diff(g, y)]])

            # Подставляем точку в матрицу Якоби
            J_at_point = J.subs({x: x0, y: y0})

            # Находим собственные значения
            eigenvalues = J_at_point.eigenvals()

            # Определяем тип точки
            point_type = classify_critical_point(eigenvalues)

            results.append({
                'point': (float(x0.evalf()), float(y0.evalf())),
                'jacobian': [[float(J_at_point[0, 0].evalf()), float(J_at_point[0, 1].evalf())],
                             [float(J_at_point[1, 0].evalf()), float(J_at_point[1, 1].evalf())]],
                'eigenvalues': [complex(str(e)) for e in eigenvalues.keys()],
                'type': point_type
            })

        return {'success': True, 'results': results}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def classify_critical_point(eigenvalues):
    """Классифицирует особую точку по собственным значениям"""
    evals = list(eigenvalues.keys())

    # Проверяем количество собственных значений
    if len(evals) < 1:
        return "Нет собственных значений"

    lambda1 = eigenvalues[evals[0]]
    lambda2 = eigenvalues[evals[1]] if len(evals) > 1 else lambda1

    # Преобразуем к комплексным числам, если необходимо
    if isinstance(lambda1, str):
        lambda1 = complex(lambda1)
    if isinstance(lambda2, str):
        lambda2 = complex(lambda2)

    # Определяем тип точки
    if lambda1.imag != 0 or lambda2.imag != 0:
        # Комплексные собственные значения
        re1 = lambda1.real
        re2 = lambda2.real

        if re1 == 0 and re2 == 0:
            return "Центр (нейтрально устойчивая)"
        elif re1 < 0 and re2 < 0:
            return "Устойчивый фокус"
        elif re1 > 0 and re2 > 0:
            return "Неустойчивый фокус"
        else:
            return "Седло-фокус"
    else:
        # Вещественные собственные значения
        lambda1 = lambda1.real
        lambda2 = lambda2.real

        if lambda1 * lambda2 < 0:
            return "Седло"
        elif lambda1 < 0 and lambda2 < 0:
            return "Устойчивый узел"
        elif lambda1 > 0 and lambda2 > 0:
            return "Неустойчивый узел"
        elif (lambda1 == 0 and lambda2 != 0) or (lambda1 != 0 and lambda2 == 0):
            return "Вырожденный узел"
        else:
            return "Не определено"


def plot_phase_portrait(f_expr, g_expr, points, x_range=(-5, 5), y_range=(-5, 5)):
    """Строит фазовый портрет"""
    try:
        # Преобразуем выражения в функции
        f_func = sp.lambdify((x, y), f_expr, 'numpy')
        g_func = sp.lambdify((x, y), g_expr, 'numpy')

        # Создаем сетку
        x_vals = np.linspace(x_range[0], x_range[1], 20)
        y_vals = np.linspace(y_range[0], y_range[1], 20)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Вычисляем производные
        U = f_func(X, Y)
        V = g_func(X, Y)

        # Нормализуем векторы для лучшего отображения
        N = np.sqrt(U ** 2 + V ** 2)
        U = U / N
        V = V / N

        # Создаем график
        plt.figure(figsize=(8, 6))
        plt.quiver(X, Y, U, V, color='blue', scale=20, width=0.005)

        # Отмечаем особые точки
        for point in points:
            x0, y0 = point['point']
            plt.plot(x0, y0, 'ro')
            plt.text(x0, y0, f"{point['type']}", fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Фазовый портрет системы')
        plt.grid(True)

        # Сохраняем график в base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return plot_data
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f_expr = request.form.get('f_expr', '')
        g_expr = request.form.get('g_expr', '')

        # Анализируем особые точки
        analysis = analyze_critical_points(f_expr, g_expr)

        if analysis['success']:
            # Строим фазовый портрет
            plot_data = plot_phase_portrait(
                sp.sympify(f_expr),
                sp.sympify(g_expr),
                analysis['results']
            )

            return render_template('index.html',
                                   results=analysis['results'],
                                   plot_data=plot_data,
                                   f_expr=f_expr,
                                   g_expr=g_expr)
        else:
            return render_template('index.html',
                                   error=analysis['error'],
                                   f_expr=f_expr,
                                   g_expr=g_expr)

    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    f_expr = data.get('f_expr', '')
    g_expr = data.get('g_expr', '')
    return jsonify(analyze_critical_points(f_expr, g_expr))


if __name__ == '__main__':
    app.run(debug=True)