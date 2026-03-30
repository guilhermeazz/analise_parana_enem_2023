import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Análise Exploratória", page_icon="📊", layout="wide")

st.title("📊 Análise Exploratória e Teste de Normalidade")
st.markdown("---")

# =====================================================================
# VERIFICAÇÃO E CARREGAMENTO DOS DADOS
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.warning("⚠️ Os dados otimizados não foram encontrados.")
    st.info("Por favor, acesse a página inicial (Home) para que o sistema realize a limpeza automática.")
    st.stop()

@st.cache_data
def carregar_dados():
    return pd.read_parquet(ARQUIVO_LIMPO)

df = carregar_dados()

# Separando os DataFrames para facilitar os cálculos e plotagens
df_pr = df[df['SG_UF_PROVA'] == 'PR']
df_br = df[df['SG_UF_PROVA'] != 'PR']

st.write("""
Nesta etapa, analisamos a forma como as notas estão distribuídas. 
O objetivo principal é comparar o desempenho do **Paraná** com o resto do **Brasil** através de métricas descritivas e verificar se as notas seguem uma **Distribuição Normal** teórica.
* **Tabela Descritiva:** Medidas de posição, dispersão e forma (Assimetria e Curtose).
* **Curvas de Densidade:** Permitem visualizar onde a maior massa de alunos se concentra.
* **QQ-Plots:** Comparam a distribuição real com a normalidade matemática. O desvio nas pontas reflete os limites do sistema TRI do ENEM e o Trimming aplicado.
""")

st.markdown("---")

# =====================================================================
# FUNÇÕES DE CÁLCULO E PLOTAGEM
# =====================================================================
def calcular_tabela_estatisticas(coluna):
    """Calcula todas as métricas estatísticas solicitadas e retorna um DataFrame formatado"""
    s_pr = df_pr[coluna].dropna()
    s_br = df_br[coluna].dropna()

    def calcular_metricas(serie):
        media = serie.mean()
        mediana = serie.median()
        # Pega a primeira moda caso existam múltiplas
        moda = serie.mode().iloc[0] if not serie.mode().empty else np.nan
        std = serie.std()
        var = serie.var()
        minimo = serie.min()
        maximo = serie.max()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        amplitude = maximo - minimo
        assimetria = serie.skew()
        curtose = serie.kurtosis()
        # Fórmula do Coeficiente de Assimetria de Pearson 2: 3 * (Média - Mediana) / Desvio Padrão
        pearson2 = 3 * (media - mediana) / std if std != 0 else np.nan
        
        return [media, mediana, moda, std, var, minimo, maximo, q1, q3, amplitude, assimetria, curtose, pearson2]

    # Realiza os cálculos
    metricas_pr = calcular_metricas(s_pr)
    metricas_br = calcular_metricas(s_br)
    
    # Calcula a diferença exata
    diferencas = [pr - br for pr, br in zip(metricas_pr, metricas_br)]

    # Formatação com 2 casas decimais e sinais corretos (+ ou -)
    def formatar_dif(val):
        if pd.isna(val): return ""
        return f"+{val:.2f}" if val > 0 else f"{val:.2f}"

    metricas_pr_str = [f"{val:.2f}" for val in metricas_pr]
    metricas_br_str = [f"{val:.2f}" for val in metricas_br]
    diferencas_str = [formatar_dif(val) for val in diferencas]

    rotulos = [
        "Média", "Mediana", "Moda", "Desvio Padrão", "Variância", 
        "Mínimo", "Máximo", "1º Quartil", "3º Quartil", "Amplitude", 
        "Assimetria", "Curtose", "Coef. Pearson 2"
    ]

    return pd.DataFrame({
        "Métrica": rotulos,
        "Paraná": metricas_pr_str,
        "Brasil": metricas_br_str,
        "Diferença": diferencas_str
    })

def plotar_densidade(coluna, titulo):
    """Gera o gráfico de curvas de densidade sobrepostas (Paraná vs Brasil)"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.kdeplot(data=df_pr[coluna], ax=ax, fill=True, label='Paraná (PR)', color='#1f77b4', alpha=0.5, linewidth=2)
    sns.kdeplot(data=df_br[coluna], ax=ax, fill=True, label='Brasil (Sem PR)', color='#ff7f0e', alpha=0.5, linewidth=2)
    
    ax.set_title(f'Distribuição de Densidade: {titulo}', weight='bold')
    ax.set_xlabel('Nota')
    ax.set_ylabel('Densidade (Frequência Relativa)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig

def plotar_qqplot(coluna, titulo):
    """Gera dois QQ-Plots lado a lado usando Scipy"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # QQ-Plot Paraná
    stats.probplot(df_pr[coluna].dropna(), dist="norm", plot=axes[0])
    axes[0].set_title('QQ-Plot: Paraná (PR)')
    axes[0].get_lines()[0].set_markerfacecolor('#1f77b4') 
    axes[0].get_lines()[0].set_markeredgecolor('#1f77b4')
    axes[0].get_lines()[1].set_color('red') 
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # QQ-Plot Brasil
    stats.probplot(df_br[coluna].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title('QQ-Plot: Brasil (Sem PR)')
    axes[1].get_lines()[0].set_markerfacecolor('#ff7f0e')
    axes[1].get_lines()[0].set_markeredgecolor('#ff7f0e')
    axes[1].get_lines()[1].set_color('red')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Aderência à Normalidade: {titulo}', weight='bold', y=1.05)
    plt.tight_layout()
    
    return fig

# =====================================================================
# RENDERIZAÇÃO DAS ABAS (TABS)
# =====================================================================
tabs = st.tabs(["Ciências da Natureza", "Ciências Humanas", "Linguagens", "Matemática", "Redação"])

colunas_nomes = [
    ('NU_NOTA_CN', 'Ciências da Natureza'),
    ('NU_NOTA_CH', 'Ciências Humanas'),
    ('NU_NOTA_LC', 'Linguagens e Códigos'),
    ('NU_NOTA_MT', 'Matemática'),
    ('NU_NOTA_REDACAO', 'Redação')
]

for i, (col, nome) in enumerate(colunas_nomes):
    with tabs[i]:
        st.subheader(f"📊 {nome}")
        
        # 1. Desenha a Tabela de Estatísticas
        st.write("**Estatística Descritiva Comparativa**")
        df_estatisticas = calcular_tabela_estatisticas(col)
        # O use_container_width=True faz a tabela esticar e ficar elegante
        st.dataframe(df_estatisticas, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True) # Dá um espaço visual
        
        # 2. Desenha a Curva de Densidade
        fig_densidade = plotar_densidade(col, nome)
        st.pyplot(fig_densidade)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Desenha os QQ-Plots
        fig_qq = plotar_qqplot(col, nome)
        st.pyplot(fig_qq)