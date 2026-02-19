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

# ── Sección 4: Modelo Keras/TensorFlow ───────────────────────────────────────
st.header("4. Equivalente con Keras / TensorFlow (Deep Learning)")

with st.expander("ℹ️ ¿Qué hace este modelo?", expanded=False):
    st.markdown(
        """
        Se entrena una **red neuronal mínima** (1 capa Dense con 1 neurona y activación lineal),
        equivalente a una regresión lineal. Los datos se normalizan con `StandardScaler` antes del entrenamiento.
        """
    )

col_keras1, col_keras2 = st.columns(2)
epochs    = col_keras1.slider("Épocas de entrenamiento", 50, 1000, 500, step=50)
lr        = col_keras2.select_slider("Learning rate (SGD)", [0.001, 0.01, 0.05, 0.1, 0.5], value=0.1)
run_keras = st.button("🚀 Entrenar modelo Keras", type="primary")

if run_keras:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input

        X_np = df[["area"]].values.astype(float)
        y_np = df[["price"]].values.astype(float)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_np)
        y_scaled = scaler_y.fit_transform(y_np)

        model = Sequential([
            Input(shape=(1,)),
            Dense(1, activation="linear"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss="mse",
        )

        with st.spinner(f"Entrenando por {epochs} épocas..."):
            history = model.fit(X_scaled, y_scaled, epochs=epochs, verbose=0)

        # Métricas
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred_keras  = scaler_y.inverse_transform(y_pred_scaled)
        mse_k  = mean_squared_error(y_np, y_pred_keras)
        rmse_k = np.sqrt(mse_k)
        r2_k   = r2_score(y_np, y_pred_keras)

        ck1, ck2, ck3 = st.columns(3)
        ck1.metric("MSE (Keras)", f"{mse_k:,.2f}")
        ck2.metric("RMSE (Keras)", f"{rmse_k:,.2f}")
        ck3.metric("R² (Keras)", f"{r2_k:.4f}")

        # Curva de pérdida
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        ax4.plot(history.history["loss"], color="#FF6D00", linewidth=1.5)
        ax4.set_xlabel("Épocas", fontsize=11)
        ax4.set_ylabel("Loss (MSE)", fontsize=11)
        ax4.set_title("Curva de pérdida durante el entrenamiento", fontsize=12, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig4)

        # Predicción con Keras
        st.subheader("Predicción con Keras")
        area_k = st.number_input("Área a predecir (m²)", value=3500, step=100, key="keras_pred")
        new_area_scaled = scaler_X.transform(np.array([[area_k]]))
        new_price_scaled = model.predict(new_area_scaled, verbose=0)
        new_price = scaler_y.inverse_transform(new_price_scaled)[0][0]
        st.success(f"Para **{area_k} m²** → Precio estimado (Keras): **${new_price:,.0f}**")

    except ImportError:
        st.error(
            "TensorFlow no está instalado en este entorno. "
            "Instálalo con: `pip install tensorflow`"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Workshop de Machine Learning · Regresión Lineal · Basado en el notebook original")
