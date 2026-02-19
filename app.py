import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import io

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Regresión Lineal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Regresión Lineal — Predicción de Precios")
st.markdown(
    """
El objetivo de este método es encontrar la **relación matemática** entre las entradas y las salidas.
En este caso una relación lineal: encontrar la **pendiente** y el **intercepto** de la recta que las configura.
"""
)

# ── Sidebar: datos ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración de datos")

data_source = st.sidebar.radio(
    "Fuente de datos",
    ["Datos de ejemplo", "Cargar archivo Excel (.xlsx)"],
)

DEFAULT_DATA = {
    "area":  [2600, 3000, 3200, 3600, 4000],
    "price": [550000, 565000, 610000, 680000, 725000],
}

if data_source == "Cargar archivo Excel (.xlsx)":
    uploaded = st.sidebar.file_uploader("Sube tu archivo data.xlsx", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        st.sidebar.success(f"Archivo cargado: {uploaded.name}")
    else:
        st.sidebar.info("Usando datos de ejemplo mientras no se sube archivo.")
        df = pd.DataFrame(DEFAULT_DATA)
else:
    df = pd.DataFrame(DEFAULT_DATA)

# ── Sección 1: Datos ─────────────────────────────────────────────────────────
st.header("1. Datos")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Dataset")
    editable = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
    df = editable.dropna().reset_index(drop=True)

with col2:
    st.subheader("Gráfico de dispersión")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["area"], df["price"], color="#4C8BF5", s=80, edgecolors="white", linewidths=1.5)
    ax.set_xlabel("Área (m²)", fontsize=12)
    ax.set_ylabel("Precio ($)", fontsize=12)
    ax.set_title("Área vs Precio", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

# ── Sección 2: Modelo sklearn ─────────────────────────────────────────────────
st.header("2. Modelo — Scikit-learn (LinearRegression)")

if len(df) < 2:
    st.warning("Necesitas al menos 2 filas de datos para entrenar el modelo.")
    st.stop()

X = df[["area"]]
y = df["price"]

reg = linear_model.LinearRegression()
reg.fit(X, y)

pendiente   = reg.coef_[0]
intercepto  = reg.intercept_
y_pred_sk   = reg.predict(X)
mse_sk      = mean_squared_error(y, y_pred_sk)
rmse_sk     = np.sqrt(mse_sk)
r2_sk       = r2_score(y, y_pred_sk)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Pendiente (m)", f"{pendiente:.4f}")
col_b.metric("Intercepto (b)", f"{intercepto:,.2f}")
col_c.metric("R²", f"{r2_sk:.4f}")

st.markdown(f"**Ecuación:** `precio = {pendiente:.2f} × área + {intercepto:,.2f}`")

col_met1, col_met2 = st.columns(2)
col_met1.metric("MSE", f"{mse_sk:,.2f}")
col_met2.metric("RMSE", f"{rmse_sk:,.2f}")

# Gráfico con recta de regresión
st.subheader("Recta de regresión")
x_line = np.linspace(df["area"].min() * 0.95, df["area"].max() * 1.05, 200)
y_line = pendiente * x_line + intercepto

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(df["area"], df["price"], color="#4C8BF5", s=80, zorder=5,
            edgecolors="white", linewidths=1.5, label="Datos reales")
ax2.plot(x_line, y_line, color="#E8453C", linewidth=2, label="Recta de regresión")
ax2.set_xlabel("Área (m²)", fontsize=12)
ax2.set_ylabel("Precio ($)", fontsize=12)
ax2.set_title("Regresión Lineal — Sklearn", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig2)

# ── Sección 3: Predicción sklearn ─────────────────────────────────────────────
st.header("3. Predicción con Scikit-learn")

area_pred = st.slider(
    "Ingresa el área a predecir (m²)",
    min_value=int(df["area"].min() * 0.5),
    max_value=int(df["area"].max() * 1.5),
    value=3300,
    step=50,
)

precio_pred = reg.predict([[area_pred]])[0]
st.success(f"Para un área de **{area_pred} m²**, el precio estimado es **${precio_pred:,.0f}**")

# Gráfico con el punto de predicción
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.scatter(df["area"], df["price"], color="#4C8BF5", s=80, zorder=5,
            edgecolors="white", linewidths=1.5, label="Datos reales")
ax3.plot(x_line, y_line, color="#E8453C", linewidth=2, label="Recta de regresión")
ax3.scatter([area_pred], [precio_pred], color="#34A853", s=150, zorder=6,
            edgecolors="white", linewidths=1.5, marker="*", label=f"Predicción ({area_pred} m²)")
ax3.annotate(
    f"${precio_pred:,.0f}",
    xy=(area_pred, precio_pred),
    xytext=(area_pred + 50, precio_pred - 15000),
    fontsize=10, color="#1a7a3a",
    arrowprops=dict(arrowstyle="->", color="#34A853"),
)
ax3.set_xlabel("Área (m²)", fontsize=12)
ax3.set_ylabel("Precio ($)", fontsize=12)
ax3.set_title("Predicción con Regresión Lineal", fontsize=13, fontweight="bold")
ax3.legend()
ax3.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig3)

# ── Sección 4: Equivalente Keras (explicación conceptual) ────────────────────
st.header("4. Equivalente con Keras / TensorFlow")

st.info(
    "💡 **Concepto:** Una red neuronal con 1 capa `Dense(1, activation='linear')` "
    "es matemáticamente equivalente a una regresión lineal. "
    "El modelo de sklearn arriba ya realiza el mismo cálculo de forma más eficiente. "
    "Para ejecutar la versión Keras, corre el notebook original en Google Colab."
)

with st.expander("Ver código Keras equivalente"):
    st.code(
        """
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

X = df[["area"]].values
y = df[["price"]].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

model = Sequential([
    Input(shape=(1,)),
    Dense(1, activation="linear"),
])
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss="mse"
)
model.fit(X_scaled, y_scaled, epochs=500, verbose=0)

# Predicción para 3500 m²
new_area_scaled = scaler_X.transform([[3500]])
pred_scaled = model.predict(new_area_scaled)
precio = scaler_y.inverse_transform(pred_scaled)[0][0]
print(f"Precio estimado: {precio:,.0f}")
        """,
        language="python",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Workshop de Machine Learning · Regresión Lineal · Basado en el notebook original")

