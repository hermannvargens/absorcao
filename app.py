import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Hidráulica de Torres Recheadas",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Painel Interativo de Hidráulica e Dimensionamento de Torres Recheadas")
st.markdown("Modelo baseado na correlação generalizada de Eckert/GPDC e no modelo hidráulico de **Billet & Schultes**.")

# ==============================================================================
# BASE DE DADOS DOS RECHEIOS (Parâmetros de Billet & Schultes e GPDC)
# ==============================================================================
BANCO_RECHEIOS = {
    'Anéis Hiflow 50 mm (Metálico)': {
        'Fp': 16.0, 'a': 92.3, 'epsilon': 0.977, 'Ch': 0.876, 'Cp': 0.421
    },
    'Anéis Pall 50 mm (Metálico)': {
        'Fp': 20.0, 'a': 110.0, 'epsilon': 0.950, 'Ch': 1.050, 'Cp': 0.480
    },
    'Anéis Pall 25 mm (Metálico)': {
        'Fp': 56.0, 'a': 215.0, 'epsilon': 0.940, 'Ch': 1.150, 'Cp': 0.520
    },
    'Anéis Raschig 50 mm (Cerâmico)': {
        'Fp': 65.0, 'a': 112.0, 'epsilon': 0.740, 'Ch': 1.400, 'Cp': 0.750
    },
    'Anéis Raschig 25 mm (Cerâmico)': {
        'Fp': 155.0, 'a': 190.0, 'epsilon': 0.730, 'Ch': 1.550, 'Cp': 0.850
    }
}

# ------------------------------------------------------------------------------
# PROPRIEDADES FÍSICAS DOS FLUIDOS E CONSTANTES (EXEMPLO 5.3)
# ------------------------------------------------------------------------------
rho_L = 986.0        # Massa específica do líquido [kg/m³]
mu_L = 0.000631      # Viscosidade dinâmica do líquido [Pa·s]

rho_G = 1.923        # Massa específica do gás [kg/m³]
mu_G = 1.45e-5       # Viscosidade dinâmica do gás [Pa·s]
g = 9.8              # Aceleração da gravidade [m/s²]

# ==============================================================================
# FUNÇÃO HIDRÁULICA DE BILLET & SCHULTES
# ==============================================================================
def calcular_delta_P_billet(D, mG_val, mL_val, a_rec, eps_rec, Ch_rec, Cp_rec):
    """Calcula a perda de carga em Pa/m segundo Billet & Schultes."""
    if D <= 0:
        return 1e6

    area = (np.pi * (D ** 2)) / 4.0
    dp = 6.0 * (1.0 - eps_rec) / a_rec

    QG_local = mG_val / rho_G
    QL_local = mL_val / rho_L

    # Fase Líquida
    vL = QL_local / area
    ReL = (vL * rho_L) / (a_rec * mu_L)
    FrL = ((vL ** 2) * a_rec) / g

    if ReL >= 5.0:
        ah = a_rec * 0.85 * Ch_rec * (ReL ** 0.25) * (FrL ** 0.1)
    else:
        ah = a_rec * Ch_rec * (ReL ** 0.25) * (FrL ** 0.1)

    hL = ((12.0 * FrL / max(ReL, 1e-8)) ** (1.0 / 3.0)) * ((ah / a_rec) ** (2.0 / 3.0))

    # Fase Gasosa
    KW = 1.0 / (1.0 + (2.0 * dp) / (3.0 * D * (1.0 - eps_rec)))
    vG = QG_local / area
    ReG = (vG * dp * rho_G * KW) / ((1.0 - eps_rec) * mu_G)

    psi_0 = Cp_rec * ((64.0 / max(ReG, 1e-8)) + (1.8 / (max(ReG, 1e-8) ** 0.08)))
    delta_P0 = (psi_0 * a_rec * rho_G * (vG ** 2)) / (2.0 * (eps_rec ** 3) * KW)

    if (eps_rec - hL) <= 0.001:
        return 2000.0  # Afogamento completo

    delta_P = delta_P0 * ((eps_rec / (eps_rec - hL)) ** 1.5) * np.exp(ReL / 200.0)
    return min(float(delta_P), 2000.0)

# ==============================================================================
# BARRA LATERAL (CONTROLES INTERATIVOS)
# ==============================================================================
st.sidebar.header("⚙️ Parâmetros Operacionais")

recheio_selecionado = st.sidebar.selectbox(
    "Tipo de Recheio:",
    options=list(BANCO_RECHEIOS.keys()),
    index=0
)

mG_val = st.sidebar.slider(
    "Vazão de Gás $m_G$ [kg/s]:",
    min_value=0.50, max_value=5.00, value=2.202, step=0.05
)

mL_val = st.sidebar.slider(
    "Vazão de Líquido $m_L$ [kg/s]:",
    min_value=0.10, max_value=3.00, value=0.804, step=0.05
)

fracao_flood = st.sidebar.slider(
    "Fração de Inundação Alvo ($f$):",
    min_value=0.40, max_value=0.85, value=0.70, step=0.01
)

diametro_escolhido = st.sidebar.slider(
    "Diâmetro Real Selecionado $D$ [m]:",
    min_value=0.40, max_value=1.50, value=0.737, step=0.005
)

# ==============================================================================
# CÁLCULOS PRINCIPAIS
# ==============================================================================
dados_rec = BANCO_RECHEIOS[recheio_selecionado]
Fp = dados_rec['Fp']
a_rec = dados_rec['a']
eps_rec = dados_rec['epsilon']
Ch_rec = dados_rec['Ch']
Cp_rec = dados_rec['Cp']

QG = mG_val / rho_G
QL = mL_val / rho_L

X_atual = (mL_val / mG_val) * np.sqrt(rho_G / rho_L)

# Flooding (GPDC)
X_faixa = np.logspace(-2.5, 0.5, 200)
Y_flood_faixa = np.exp(-(3.5021 + 1.028 * np.log(X_faixa) + 0.11093 * (np.log(X_faixa) ** 2)))
Y_70_faixa = (0.70 ** 2) * Y_flood_faixa
Y_50_faixa = (0.50 ** 2) * Y_flood_faixa

Y_flood_op = float(np.exp(-(3.5021 + 1.028 * np.log(X_atual) + 0.11093 * (np.log(X_atual) ** 2))))
Csf = np.sqrt(Y_flood_op / (Fp * (mu_L ** 0.1)))
vGf = Csf / np.sqrt(rho_G / (rho_L - rho_G))

D_flood = np.sqrt((4.0 * QG) / (vGf * np.pi))
D_projetado = np.sqrt((4.0 * QG) / (fracao_flood * vGf * np.pi))
dP_projetado = calcular_delta_P_billet(D_projetado, mG_val, mL_val, a_rec, eps_rec, Ch_rec, Cp_rec)

area_escolhida = (np.pi * (diametro_escolhido ** 2)) / 4.0
vG_op = QG / area_escolhida
percentual_flood_real = (vG_op / vGf) * 100.0

Cs_op = vG_op * np.sqrt(rho_G / (rho_L - rho_G))
Y_op = (Cs_op ** 2) * Fp * (mu_L ** 0.1)
dP_op = calcular_delta_P_billet(diametro_escolhido, mG_val, mL_val, a_rec, eps_rec, Ch_rec, Cp_rec)

# ==============================================================================
# PAINEL DE MÉTRICAS E STATUS
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Parâmetro de Fluxo (X)", f"{X_atual:.4f}")
col2.metric(f"D Projeto ({fracao_flood*100:.0f}% Flood)", f"{D_projetado:.3f} m", f"{dP_projetado:.1f} Pa/m")
col3.metric("D Selecionado", f"{diametro_escolhido:.3f} m", f"{percentual_flood_real:.1f}% do Flood")
col4.metric("ΔP Operacional", f"{dP_op:.1f} Pa/m")

if dP_op >= 600 or percentual_flood_real >= 90:
    st.error("⚠️ **ALERTA**: Coluna em regime de AFOGAMENTO ou com perda de carga excessiva (> 600 Pa/m ou > 90% do Flooding)!")
elif 200 <= dP_op <= 400:
    st.success("✅ **DIAGNÓSTICO**: Operação dentro da FAIXA ÓTIMA recomendada para absorção (200 a 400 Pa/m).")
else:
    st.warning("ℹ️ **DIAGNÓSTICO**: Coluna fora da faixa recomendada (superdimensionada ou sobrecarregada).")

# ==============================================================================
# GRÁFICOS
# ==============================================================================
D_min_plot = max(0.35, D_flood * 0.95)
D_max_plot = max(1.40, D_projetado * 1.35)
D_faixa = np.linspace(D_min_plot, D_max_plot, 100)
dP_faixa = [calcular_delta_P_billet(d, mG_val, mL_val, a_rec, eps_rec, Ch_rec, Cp_rec) for d in D_faixa]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# --- GRÁFICO 1: GPDC ---
ax1.loglog(X_faixa, Y_flood_faixa, 'r-', lw=2.2, label='100% Inundação (Flooding)')
ax1.loglog(X_faixa, Y_70_faixa, 'b--', lw=1.8, label='70% da Inundação')
ax1.loglog(X_faixa, Y_50_faixa, 'g:', lw=1.8, label='50% da Inundação')
ax1.plot(X_atual, Y_op, 'o', color='purple', markersize=9, zorder=5,
         label=f'Operação ({percentual_flood_real:.1f}% Flood)')
ax1.axvline(X_atual, color='gray', linestyle=':', alpha=0.6)

ax1.set_title('Diagrama GPDC - Curvas de Capacidade', fontsize=12, fontweight='bold')
ax1.set_xlabel(r'Parâmetro de Fluxo $X = (m_L / m_G) \cdot (\rho_G / \rho_L)^{0.5}$', fontsize=10)
ax1.set_ylabel(r'Parâmetro de Capacidade $Y$', fontsize=10)
ax1.set_xlim(0.005, 1.0)
ax1.set_ylim(0.001, 1.0)
ax1.grid(True, which="both", ls="--", alpha=0.35)
ax1.legend(loc='lower left', frameon=True, fontsize=9)

# --- GRÁFICO 2: JANELA OPERACIONAL ---
ax2.plot(D_faixa, dP_faixa, color='#1f77b4', lw=2.5, label=r'Curva Billet & Schultes $\Delta P(D)$')
ax2.axhspan(200, 400, color='green', alpha=0.18, label='Faixa Recomendada (200 - 400 Pa/m)')
ax2.axhline(300, color='darkgreen', linestyle='--', lw=1.2, label='Alvo de Projeto (300 Pa/m)')
ax2.axvline(D_flood, color='red', linestyle='--', lw=1.5, label=f'Flooding 100% ($D = {D_flood:.3f}$ m)')
ax2.axvline(D_projetado, color='orange', linestyle='-.', lw=2.0,
            label=f'Projeto f={fracao_flood*100:.0f}% ($D = {D_projetado:.3f}$ m)')

cor_ponto = 'purple' if 200 <= dP_op <= 400 else ('red' if dP_op > 600 else 'brown')
ax2.plot(diametro_escolhido, min(dP_op, 1150), 'o', color=cor_ponto, markersize=10, zorder=5,
         label=f'D Selecionado: {diametro_escolhido:.3f} m ($\Delta P = {dP_op:.0f}$ Pa/m)')

ax2.set_title('Janela Hidráulica Operacional', fontsize=12, fontweight='bold')
ax2.set_xlabel(r'Diâmetro Interno da Coluna $D$ [m]', fontsize=10)
ax2.set_ylabel(r'Perda de Carga do Gás $\Delta P / Z$ [Pa/m]', fontsize=10)
ax2.set_xlim(D_min_plot, D_max_plot)
ax2.set_ylim(0, 1200)
ax2.grid(True, ls=":", alpha=0.5)
ax2.legend(loc='upper right', frameon=True, fontsize=8.5)

plt.tight_layout()
st.pyplot(fig)

# ==============================================================================
# DETALHAMENTO EXPANSÍVEL
# ==============================================================================
with st.expander("📋 Ver Dados Numéricos Detalhados"):
    st.markdown(f"""
    - **Recheio**: {recheio_selecionado} ($F_p = {Fp}\\text{{ ft}}^{{-1}}$, $a = {a_rec}\\text{{ m}}^2/\\text{{m}}^3$, $\\varepsilon = {eps_rec}$)
    - **Velocidade de Inundação ($v_{{Gf}}$)**: `{vGf:.3f} m/s`
    - **Diâmetro Mínimo de Inundação ($D_{{flood}}$)**: `{D_flood:.3f} m`
    - **Velocidade Superficial Real ($v_G$)**: `{vG_op:.3f} m/s`
    - **Vazão Volumétrica do Gás ($Q_G$)**: `{QG:.4f} m³/s` | **Líquido ($Q_L$)**: `{QL*1000:.3f} L/s`
    """)
