import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# ==========================================
# 0. CONFIGURACIÓN DE STREAMLIT
# ==========================================
st.set_page_config(page_title="PdM Gasoil Paysandú", layout="wide")
st.title("🔋 Predictor de Mantenimiento Fiscal (Physics-Informed ML)")
st.markdown("Este modelo evalúa el riesgo de deriva en caudalímetros combinando el **desgaste mecánico (Litros)** y la **deposición de parafinas (Días estancado)**.")

# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
archivo = 'datos_Paysandu.xlsx'

try:
    # Lectura robusta forzando las cabeceras
    raw = pd.read_excel(archivo, engine="openpyxl", header=None)
    headers = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = headers
    df.columns = df.columns.str.strip() # Limpia espacios extraños
    df = df.dropna(how="all")

    # Conversión numérica
    df['Millones_Litros'] = pd.to_numeric(df['LITROS DE GASOIL'], errors='coerce') / 1_000_000
    df['Dias_Inactividad'] = pd.to_numeric(df['DÍAS DESDE ÚLTIMA CALIBRACIÓN'], errors='coerce')
    df['DERIVA'] = pd.to_numeric(df['DERIVA'], errors='coerce').astype(int)
    
    # Nos aseguramos de no tener nulos
    df = df.dropna(subset=['Millones_Litros', 'Dias_Inactividad', 'DERIVA'])

    # ¡AQUÍ SE CREAN X e Y! (Lo que causaba el NameError)
    X = df[['Millones_Litros', 'Dias_Inactividad']]
    y_clasificacion = df['DERIVA']

except Exception as e:
    st.error(f"Error al cargar o procesar los datos de Paysandú. Verifica que el archivo exista en GitHub. Detalle: {e}")
    st.stop() # Detiene la app limpiamente si falla la lectura

# ==========================================
# 2. MODELO FÍSICO (PIML)
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_arr = y_clasificacion.values

# Pesos para datos desbalanceados
w0 = len(y_arr) / (2 * np.sum(y_arr == 0))
w1 = len(y_arr) / (2 * np.sum(y_arr == 1))
sample_weights = np.where(y_arr == 1, w1, w0)

def weighted_log_loss(coefs, X_mat, y, weights):
    z = coefs[0] + np.dot(X_mat, coefs[1:])
    p = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(weights * (y * np.log(p) + (1 - y) * np.log(1 - p)))

# Restricciones físicas: Litros y Días SIEMPRE suman desgaste
bounds = [(None, None), (0, None), (0.0001, None)]
res = minimize(weighted_log_loss, x0=np.zeros(3), args=(X_scaled, y_arr, sample_weights), bounds=bounds)

intercept_scaled, beta1_scaled, beta2_scaled = res.x
mu_1, mu_2 = scaler.mean_
sigma_1, sigma_2 = scaler.scale_

# Deshacer el escalado para usar los números reales
beta1_orig = beta1_scaled / sigma_1
beta2_orig = beta2_scaled / sigma_2
intercept_orig = intercept_scaled - (beta1_scaled * mu_1 / sigma_1) - (beta2_scaled * mu_2 / sigma_2)

def predict_proba_fisica(X_orig):
    z = intercept_orig + X_orig[:, 0] * beta1_orig + X_orig[:, 1] * beta2_orig
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# ==========================================
# 3. INTERFAZ WEB: MÉTRICAS
# ==========================================
st.subheader("Parámetros Físicos Aprendidos por el Modelo")
col1, col2 = st.columns(2)
col1.metric("Peso del Desgaste (Litros)", f"{beta1_orig:.4f}")
col2.metric("Peso del Estancamiento (Días)", f"{beta2_orig:.4f}")

# ==========================================
# 4. GRÁFICO DEL MAPA DE RIESGO
# ==========================================
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

ax.set_title('Mapa de Riesgo de Calibración Fiscal', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel(f'Volumen Bombeado (Millones de Litros)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'Tiempo desde última calibración (Días)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=12)

plt.tight_layout()

# Renderizado final en la web
st.pyplot(fig)
