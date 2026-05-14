import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import pandas as pd

# =============================================================
# CONFIGURACIÓN DE STREAMLIT
# =============================================================
st.set_page_config(page_title="Dashboard Proyección Paysandú", layout="wide")

st.sidebar.header("⚙️ Parámetros de Simulación")
HOY_date = st.sidebar.date_input("Fecha de Referencia (HOY)", datetime(2026, 5, 14))
HOY = datetime(HOY_date.year, HOY_date.month, HOY_date.day)

EMP = st.sidebar.number_input("Error Máximo Permitido (EMP %)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

st.title("📊 Proyección de Descalibración – Medidores Volumétricos Gasoil")
st.markdown(f"**Paysandú 2025–2027** | Ref.: {HOY.strftime('%d/%m/%Y')}")

# =============================================================
# DATOS REALES - MEDIDORES PAYSANDÚ 2025
# =============================================================
series     = ["UF151335\nBrazo 2", "UF151327\nBrazo 5",
              "UF151340\nBrazo 6", "UF151354\nBrazo 9",
              "UF151330\nBrazo 10"]
series_id  = ["B2","B5","B6","B9","B10"]

fechas_cal = [
    datetime(2025, 11, 14),   # Brazo 2
    datetime(2025, 11, 15),   # Brazo 5
    datetime(2025, 11, 15),   # Brazo 6
    datetime(2025, 11, 15),   # Brazo 9
    datetime(2025, 11, 16),   # Brazo 10
]

errores    = np.array([-0.04,  0.08,  0.00, -0.08, -0.16])
litros     = np.array([7_728_205, 12_413_892, 8_120_805, 9_319_617, 25_252_583])
deriva     = np.array([0, 1, 0, 1, 1])
dias       = np.array([329, 329, 329, 329, 330])

media      = np.mean(errores)
desvio     = np.std(errores, ddof=1)

# Cálculos de proyección
litros_dia = litros / dias
tasa_dia   = errores / dias
tasa_mlit  = np.where(litros > 0, errores / (litros / 1e6), 0)

resultados = []
for i in range(len(series_id)):
    serie   = series_id[i]
    error   = errores[i]
    t_dia   = tasa_dia[i]
    t_mlit  = tasa_mlit[i]
    ldia    = litros_dia[i]
    f_cal   = fechas_cal[i]
    dias_tr = max((HOY - f_cal).days, 1)

    error_hoy = error + t_dia * dias_tr

    if t_dia > 0:
        margen = EMP - error_hoy
    elif t_dia < 0:
        margen = -EMP - error_hoy
    else:
        margen = EMP - abs(error_hoy)

    if abs(t_dia) > 1e-9:
        dias_hasta_emp = abs(margen / t_dia)
        fecha_emp      = HOY + timedelta(days=dias_hasta_emp)
        litros_hasta   = dias_hasta_emp * ldia
    else:
        dias_hasta_emp = float('inf')
        fecha_emp      = None
        litros_hasta   = float('inf')

    sigma_proy = desvio * np.sqrt(dias_tr / np.mean(dias))
    p_actual   = (1 - stats.norm.cdf( EMP, error_hoy, sigma_proy) +
                      stats.norm.cdf(-EMP, error_hoy, sigma_proy)) * 100

    resultados.append({
        "serie":           serie,
        "error_orig":      error,
        "error_hoy":       error_hoy,
        "tasa_dia":        t_dia,
        "litros_dia":      ldia,
        "dias_hasta_emp":  dias_hasta_emp,
        "fecha_emp":       fecha_emp,
        "litros_hasta":    litros_hasta,
        "p_actual":        p_actual
    })

# =============================================================
# ESTILO Y GRÁFICAS MATPLOTLIB
# =============================================================
COLOR_FONDO = '#1e1e2e'
COLOR_PANEL = '#2a2a3e'
COLOR_TEXTO = '#e0e0f0'
COLOR_GRID  = '#3a3a5e'
COLOR_EMP   = '#ff4757'
COLOR_MEDIA = '#2ed573'

def estilo_ax(ax, titulo):
    ax.set_facecolor(COLOR_PANEL)
    ax.set_title(titulo, color=COLOR_TEXTO, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=COLOR_TEXTO, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXTO)
    ax.yaxis.label.set_color(COLOR_TEXTO)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.7)

fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor(COLOR_FONDO)
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30, top=0.95, bottom=0.05)

etiq_cortas = ["Brazo 2", "Brazo 5", "Brazo 6", "Brazo 9", "Brazo 10"]
palette = ['#2ed573','#ffd32a','#1e90ff','#ff6b81','#a29bfe']

# --- GRÁFICA 1 ---
ax1 = fig.add_subplot(gs[0, :])
estilo_ax(ax1, "① Proyección del Error de Verificación en el Tiempo por Medidor")

for i, r in enumerate(resultados):
    f0 = fechas_cal[i]
    t_fin = min(r["dias_hasta_emp"], (datetime(2027, 5, 31) - f0).days)
    t_fin = min(t_fin, 900)
    t_arr = np.linspace(0, t_fin, 300)
    e_arr = errores[i] + r["tasa_dia"] * t_arr
    d_arr = [f0 + timedelta(days=float(t)) for t in t_arr]

    ax1.plot(d_arr, e_arr, color=palette[i], linewidth=2.5, label=f'{etiq_cortas[i]} ({errores[i]:+.2f}%)')
    ax1.scatter([f0], [errores[i]], color=palette[i], s=90, edgecolors='white', linewidth=1)

    if r["fecha_emp"] and r["fecha_emp"] <= datetime(2027, 5, 31):
        emp_val = EMP if r["tasa_dia"] > 0 else -EMP
        ax1.scatter([r["fecha_emp"]], [emp_val], marker='X', s=150, color=COLOR_EMP, edgecolors='white')
        ax1.annotate(f'  {etiq_cortas[i]}\n  {r["fecha_emp"].strftime("%b-%Y")}', 
                     xy=(r["fecha_emp"], emp_val), fontsize=8, color=palette[i], xytext=(8, -12), textcoords='offset points')

ax1.axhline( EMP, color=COLOR_EMP, linestyle='--', linewidth=2, label=f'+EMP = +{EMP}%')
ax1.axhline(-EMP, color=COLOR_EMP, linestyle='--', linewidth=2, label=f'–EMP = –{EMP}%')
ax1.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
ax1.axvline(HOY, color='#ffeaa7', linestyle='-.', linewidth=2, label='Día Referencia (HOY)')
ax1.text(HOY, EMP * 1.05, ' HOY', color='#ffeaa7', fontsize=9, va='bottom')
ax1.axhspan(-EMP, EMP, alpha=0.07, color=COLOR_MEDIA)

ax1.set_ylabel("Error proyectado (%)")
ax1.set_ylim(-max(0.40, EMP*1.5), max(0.40, EMP*1.5))
ax1.legend(fontsize=9, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXTO, edgecolor=COLOR_GRID, loc='upper left', ncol=4)

# --- GRÁFICA 2 ---
ax2 = fig.add_subplot(gs[1, 0])
estilo_ax(ax2, "② Días Restantes hasta Alcanzar el EMP")
dias_rest = [900 if r["dias_hasta_emp"] == float('inf') else max((r["fecha_emp"] - HOY).days, 0) for r in resultados]
colores_dias = ['#2ecc71' if d >= 365 else '#f39c12' if d >= 180 else '#e74c3c' for d in dias_rest]

bars2 = ax2.barh(etiq_cortas[::-1], dias_rest[::-1], color=colores_dias[::-1], edgecolor='white', height=0.5)
ax2.axvline(365, color='#f39c12', linestyle='--', linewidth=1.5, label='1 año')
ax2.axvline(180, color=COLOR_EMP, linestyle=':', linewidth=1.5, label='6 meses')

for bar, d, r in zip(bars2, dias_rest[::-1], resultados[::-1]):
    lbl = "Estable" if d >= 900 else f'{d} días ({r["fecha_emp"].strftime("%d/%m/%Y")})'
    ax2.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2, lbl, va='center', color=COLOR_TEXTO, fontsize=9)

ax2.set_xlabel("Días restantes")
ax2.set_xlim(0, max(1150, max(dias_rest)*1.2))
ax2.legend(fontsize=8, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXTO, edgecolor=COLOR_GRID)

# --- GRÁFICA 3 ---
ax3 = fig.add_subplot(gs[1, 1])
estilo_ax(ax3, "③ Millones de Litros Restantes hasta EMP")
mlit_rest = [50.0 if r["litros_hasta"] == float('inf') else max(r["litros_hasta"] - (HOY - fechas_cal[i]).days * r["litros_dia"], 0) / 1e6 for i, r in enumerate(resultados)]
colores_lit = ['#2ecc71' if m >= 10 else '#f39c12' if m >= 5 else '#e74c3c' for m in mlit_rest]

bars3 = ax3.barh(etiq_cortas[::-1], mlit_rest[::-1], color=colores_lit[::-1], edgecolor='white', height=0.5)
ax3.axvline(10, color='#f39c12', linestyle='--', linewidth=1.5, label='10 M litros')
ax3.axvline(5,  color=COLOR_EMP, linestyle=':', linewidth=1.5, label='5 M litros')

for bar, m in zip(bars3, mlit_rest[::-1]):
    lbl = "Estable" if m >= 50 else f'{m:.1f} M L'
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, lbl, va='center', color=COLOR_TEXTO, fontsize=9, fontweight='bold')

ax3.set_xlabel("Millones de litros restantes")
ax3.set_xlim(0, 62)
ax3.legend(fontsize=8, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXTO, edgecolor=COLOR_GRID)

# --- GRÁFICA 4 ---
ax4 = fig.add_subplot(gs[2, 0])
estilo_ax(ax4, "④ Probabilidad de Descalibración Actual (%)")
probs = [r["p_actual"] for r in resultados]
colores_p = ['#2ecc71' if p < 5 else '#f39c12' if p < 20 else '#e74c3c' for p in probs]

bars4 = ax4.bar(etiq_cortas, probs, color=colores_p, edgecolor='white', width=0.55)
ax4.axhline(5,  color='#f39c12', linestyle='--', linewidth=1.5, label='Alerta (5%)')
ax4.axhline(20, color=COLOR_EMP, linestyle=':', linewidth=1.5, label='Crítico (20%)')

for bar, p in zip(bars4, probs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{p:.2f}%', ha='center', va='bottom', color=COLOR_TEXTO, fontsize=10, fontweight='bold')

ax4.set_ylabel("P(descalibración) %")
ax4.set_ylim(0, max(probs) * 1.35 + 5)
ax4.legend(fontsize=8, facecolor=COLOR_PANEL, labelcolor=COLOR_TEXTO, edgecolor=COLOR_GRID)

# --- RESUMEN (Reemplazando Tabla estática por st.dataframe) ---
st.pyplot(fig)

st.markdown("### ⑤ Tabla Resumen de Proyecciones")
df_resumen = pd.DataFrame([
    {
        "Medidor": r["serie"].replace('\n', ' '),
        "Error Proyectado HOY (%)": f"{r['error_hoy']:+.4f}",
        "Tasa de Deriva (%/día)": f"{r['tasa_dia']*1000:+.4f} ‰",
        "Fecha est. EMP": r["fecha_emp"].strftime("%d/%m/%Y") if r["fecha_emp"] else "Estable",
        "Días Restantes": int((r['fecha_emp']-HOY).days) if r["fecha_emp"] else "N/A",
        "P(descalibración) %": f"{r['p_actual']:.2f}%"
    }
    for r in resultados
])

st.dataframe(df_resumen, use_container_width=True)
