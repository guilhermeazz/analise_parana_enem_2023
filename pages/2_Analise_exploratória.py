import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache_graficos')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Análise Exploratória", page_icon="📊", layout="wide")
st.title("📊 Análise Exploratória e Teste de Normalidade")
st.markdown("---")

# =====================================================================
# MOTOR DE CACHE DE GRÁFICOS (JSON)
# =====================================================================
def obter_grafico_cache(nome_arquivo, funcao_geradora):
    caminho = os.path.join(CACHE_DIR, nome_arquivo)
    if os.path.exists(caminho):
        return pio.read_json(caminho)
    
    fig = funcao_geradora()
    pio.write_json(fig, caminho)
    return fig

# =====================================================================
# VERIFICAÇÃO E CARREGAMENTO DOS DADOS
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.warning("⚠️ Os dados otimizados não foram encontrados.")
    st.stop()

@st.cache_data
def carregar_dados():
    return pd.read_parquet(ARQUIVO_LIMPO)

df = carregar_dados()

# =====================================================================
# FUNÇÕES DE PRÉ-CÁLCULO E PLOTAGEM (USANDO 100% DOS DADOS)
# =====================================================================
@st.cache_data
def calcular_tabela_estatisticas(coluna):
    """Calcula usando 100% dos dados para exatidão milimétrica"""
    s_pr = df[df['SG_UF_PROVA'] == 'PR'][coluna].dropna()
    s_br = df[df['SG_UF_PROVA'] != 'PR'][coluna].dropna()

    def calc(serie):
        return [serie.mean(), serie.median(), serie.mode().iloc[0] if not serie.mode().empty else np.nan,
                serie.std(), serie.var(), serie.min(), serie.max(), 
                serie.quantile(0.25), serie.quantile(0.75), serie.max() - serie.min(), 
                serie.skew(), serie.kurtosis(), 
                3 * (serie.mean() - serie.median()) / serie.std() if serie.std() != 0 else np.nan]

    metricas_pr, metricas_br = calc(s_pr), calc(s_br)
    
    return pd.DataFrame({
        "Métrica": ["Média", "Mediana", "Moda", "Desvio Padrão", "Variância", "Mínimo", "Máximo", "1º Quartil", "3º Quartil", "Amplitude", "Assimetria", "Curtose", "Coef. Pearson 2"],
        "Paraná": [f"{v:.2f}" for v in metricas_pr],
        "Brasil": [f"{v:.2f}" for v in metricas_br],
        "Diferença": [f"+{pr-br:.2f}" if pr-br > 0 else f"{pr-br:.2f}" for pr, br in zip(metricas_pr, metricas_br)]
    })

def gerar_densidade_precalculada(coluna, titulo):
    """Calcula a curva no backend usando 100% dos dados e desenha com apenas 500 coordenadas"""
    fig = go.Figure()
    
    # Adicionamos uma quarta variável com a cor RGBA (O último número, 0.4, é a transparência do preenchimento)
    config_cores = [
        ('PR', 'Paraná (PR)', '#1f77b4', 'rgba(31, 119, 180, 0.4)'), 
        ('BR', 'Brasil (Sem PR)', '#ff7f0e', 'rgba(255, 127, 14, 0.4)')
    ]
    
    for regiao, label, cor_linha, cor_preenchimento in config_cores:
        data = df[df['SG_UF_PROVA'] == regiao if regiao == 'PR' else df['SG_UF_PROVA'] != 'PR'][coluna].dropna()
        
        if not data.empty:
            # PRÉ-CÁLCULO MATEMÁTICO: KDE baseado em 100% da base real
            kde = stats.gaussian_kde(data)
            # Cria 500 pontos X uniformemente espaçados
            x_vals = np.linspace(data.min(), data.max(), 500)
            # Descobre o Y exato para esses 500 pontos
            y_vals = kde(x_vals)
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines', 
                fill='tozeroy', name=label, 
                line=dict(color=cor_linha, width=2.5), # Linha sólida e um pouco mais grossa
                fillcolor=cor_preenchimento           # Transparência apenas na "tinta" do gráfico
            ))
            
    fig.update_layout(
        title=f"Distribuição de Densidade: {titulo}",
        xaxis_title="Nota", yaxis_title="Densidade", margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def gerar_qqplot_precalculado(coluna, titulo):
    """Calcula quantis exatos usando 100% dos dados, plotando apenas os limiares representativos"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Paraná (PR)", "Brasil (Sem PR)"))
    
    for i, (regiao, cor) in enumerate([('PR', '#1f77b4'), ('BR', '#ff7f0e')]):
        data = df[df['SG_UF_PROVA'] == regiao if regiao == 'PR' else df['SG_UF_PROVA'] != 'PR'][coluna].dropna()
        
        if not data.empty:
            # PRÉ-CÁLCULO: Extrai 1.000 percentis exatos (do 0.1% ao 99.9%) usando 100% da base
            percentis = np.linspace(0.1, 99.9, 1000)
            data_quantiles = np.percentile(data, percentis)
            norm_quantiles = stats.norm.ppf(percentis / 100.0)
            
            # Cálculo da reta teórica vermelha (baseado em Q1 e Q3)
            q1_d, q3_d = np.percentile(data, [25, 75])
            q1_n, q3_n = stats.norm.ppf([0.25, 0.75])
            slope = (q3_d - q1_d) / (q3_n - q1_n)
            intercept = q1_d - slope * q1_n
            
            fig.add_trace(go.Scatter(x=norm_quantiles, y=data_quantiles, mode='markers', marker=dict(color=cor, size=4, opacity=0.6), name=regiao), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=norm_quantiles, y=slope*norm_quantiles + intercept, mode='lines', line=dict(color='red', width=2), showlegend=False), row=1, col=i+1)
            
    fig.update_layout(title=f"Aderência à Normalidade (QQ-Plot): {titulo}", showlegend=False, height=400, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# =====================================================================
# RENDERIZAÇÃO DAS ABAS (TABS)
# =====================================================================
st.write("Nesta etapa, analisamos a forma como as notas estão distribuídas, utilizando **100% da base tratada** e processamento vetorial para máxima precisão estatística e interatividade na web.")

tabs = st.tabs(["Ciências da Natureza", "Ciências Humanas", "Linguagens", "Matemática", "Redação"])
colunas_nomes = [('NU_NOTA_CN', 'Ciências da Natureza'), ('NU_NOTA_CH', 'Ciências Humanas'), ('NU_NOTA_LC', 'Linguagens e Códigos'), ('NU_NOTA_MT', 'Matemática'), ('NU_NOTA_REDACAO', 'Redação')]

for i, (col, nome) in enumerate(colunas_nomes):
    with tabs[i]:
        st.subheader(f"📊 {nome}")
        
        st.write("**Estatística Descritiva Comparativa**")
        st.dataframe(calcular_tabela_estatisticas(col), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(obter_grafico_cache(f"densidade_precalc_{col}.json", lambda c=col, n=nome: gerar_densidade_precalculada(c, n)), use_container_width=True)
        with c2:
            st.plotly_chart(obter_grafico_cache(f"qqplot_precalc_{col}.json", lambda c=col, n=nome: gerar_qqplot_precalculado(c, n)), use_container_width=True)