import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# 1. Recuperar X e y y estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_arr = y_clasificacion.values

# 2. Asignar pesos a las clases para datos desbalanceados (class_weight='balanced')
w0 = len(y_arr) / (2 * np.sum(y_arr == 0))
w1 = len(y_arr) / (2 * np.sum(y_arr == 1))
sample_weights = np.where(y_arr == 1, w1, w0)

# 3. Función de costo Log-Loss
def weighted_log_loss(coefs, X_mat, y, weights):
    z = coefs[0] + np.dot(X_mat, coefs[1:])
    p = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(weights * (y * np.log(p) + (1 - y) * np.log(1 - p)))

# 4. RESTRICCIONES FÍSICAS (Physics-Informed)
# Forzamos a que el modelo le asigne peso > 0 a los Litros y a los Días
# Usamos un valor mínimo muy chiquito para los días (0.0001) como pide la física del problema
bounds = [(None, None), (0, None), (0.0001, None)]

res = minimize(weighted_log_loss, x0=np.zeros(3), args=(X_scaled, y_arr, sample_weights), bounds=bounds)

intercept_scaled, beta1_scaled, beta2_scaled = res.x
mu_1, mu_2 = scaler.mean_
sigma_1, sigma_2 = scaler.scale_

# 5. Convertir a coeficientes originales (sin escalar)
beta1_orig = beta1_scaled / sigma_1
beta2_orig = beta2_scaled / sigma_2
intercept_orig = intercept_scaled - (beta1_scaled * mu_1 / sigma_1) - (beta2_scaled * mu_2 / sigma_2)

def predict_proba_fisica(X_orig):
    z = intercept_orig + X_orig[:, 0] * beta1_orig + X_orig[:, 1] * beta2_orig
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# 6. Graficar Mapa de Riesgo
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

x_min, x_max = 0, X['Millones_Litros'].max() * 1.15
y_min, y_max = 0, X['Dias_Inactividad'].max() * 1.15
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
malla_df = np.c_[xx.ravel(), yy.ravel()]

Z = predict_proba_fisica(malla_df).reshape(xx.shape)
cmap = sns.diverging_palette(130, 10, as_cmap=True)
contourf = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 15), cmap=cmap, alpha=0.3)
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Probabilidad de Deriva (%)', fontsize=12, fontweight='bold')

contour = ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=4, linestyles='--')
ax.clabel(contour, inline=True, fontsize=14, fmt={0.5: ' P = 50% '})

ax.scatter(X[y_clasificacion == 0]['Millones_Litros'], X[y_clasificacion == 0]['Dias_Inactividad'], 
           c='#2ca02c', s=150, edgecolors='black', linewidths=2, label='Calibración Exitosa (0)')
ax.scatter(X[y_clasificacion == 1]['Millones_Litros'], X[y_clasificacion == 1]['Dias_Inactividad'], 
           c='#d62728', s=150, edgecolors='black', linewidths=2, label='Fallo / Deriva (1)')

# 7. Corte en P=50%
corte_x = -intercept_orig / beta1_orig
ax.plot([corte_x], [0], marker='X', markersize=12, color='black')
ax.annotate(f'P=50% corta en:\n{corte_x:.2f} Millones de L', 
            xy=(corte_x, 0), xytext=(corte_x + 2, y_max * 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5),
            fontsize=12, fontweight='bold')

ax.text(x_max * 0.05, y_max * 0.9, 'ZONA SEGURA\n(Bajo Riesgo)', 
        fontsize=14, fontweight='bold', bbox=dict(facecolor='#2ca02c', alpha=0.3, boxstyle='round,pad=0.5'))
ax.text(x_max * 0.7, y_max * 0.9, 'ZONA DE RIESGO\n(Alto Riesgo)', 
        fontsize=14, fontweight='bold', bbox=dict(facecolor='#d62728', alpha=0.3, boxstyle='round,pad=0.5'))

ax.set_title('Mapa de Riesgo de Calibración - Modelo Physics-Informed', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel(f'Volumen Bombeado (Millones de Litros)\nPeso: {beta1_orig:.4f}', fontsize=14, fontweight='bold')
ax.set_ylabel(f'Tiempo desde última calibración (Días)\nPeso: {beta2_orig:.4f}', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=12)

plt.tight_layout()
plt.show()
