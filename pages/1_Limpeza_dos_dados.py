import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache_graficos')
ARQUIVO_BRUTO = os.path.join(DATA_DIR, 'enem2023.parquet')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Limpeza de Dados", page_icon="🧹", layout="wide")

st.title("🧹 Processo de Limpeza e Preparação dos Dados")
st.markdown("---")

# =====================================================================
# MOTOR DE CACHE (JSON)
# =====================================================================
def obter_grafico_cache(nome_arquivo, funcao_geradora):
    caminho = os.path.join(CACHE_DIR, nome_arquivo)
    if os.path.exists(caminho):
        return pio.read_json(caminho)
    
    fig = funcao_geradora()
    pio.write_json(fig, caminho)
    return fig

# =====================================================================
# CARREGAMENTO DOS DADOS (PARA CÁLCULOS DE MÉTRICAS)
# =====================================================================
@st.cache_data
def carregar_dados_basicos():
    colunas = ['SG_UF_PROVA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    df_b = pd.read_parquet(ARQUIVO_BRUTO, columns=colunas)
    df_l = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    return df_b, df_l

df_bruto, df_limpo = carregar_dados_basicos()

# =====================================================================
# 1. TRATAMENTO DE DADOS FALTANTES
# =====================================================================
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

# Cálculos dinâmicos para as métricas
faltantes_pr = int(df_bruto[df_bruto['SG_UF_PROVA'] == 'PR'][colunas_notas].isnull().any(axis=1).sum())
faltantes_br = int(df_bruto[df_bruto['SG_UF_PROVA'] != 'PR'][colunas_notas].isnull().any(axis=1).sum())

st.header("1. Tratamento de Dados Faltantes")
st.write("A base de dados bruta do ENEM possui um elevado índice de abstenções, o que gera dados nulos (NaN) nas colunas de notas. Para garantir a integridade da modelagem estatística, todos os alunos com notas em branco foram removidos da nossa amostra.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tamanho Original da Base", f"{len(df_bruto):,}".replace(',', '.'))
c2.metric("Tamanho Após Limpeza", f"{len(df_limpo):,}".replace(',', '.'))
c3.metric("Faltantes (Brasil sem PR)", f"{faltantes_br:,}".replace(',', '.'))
c4.metric("Faltantes (Paraná)", f"{faltantes_pr:,}".replace(',', '.'))

st.markdown("---")

# =====================================================================
# 2. TRATAMENTO DE OUTLIERS (BOXPLOTS)
# =====================================================================
st.header("2. Tratamento de Outliers (Boxplots)")
st.write("""
Para evitar distorções nos modelos preditivos e garantir a estabilização da variância estocástica, optámos pela técnica de **Truncamento (Trimming)** baseada no Intervalo Interquartil (IQR). 
""")

tabs = st.tabs(["Ciências da Natureza", "Ciências Humanas", "Linguagens", "Matemática", "Redação"])

colunas_nomes = [
    ('NU_NOTA_CN', 'Ciências da Natureza'),
    ('NU_NOTA_CH', 'Ciências Humanas'),
    ('NU_NOTA_LC', 'Linguagens e Códigos'),
    ('NU_NOTA_MT', 'Matemática'),
    ('NU_NOTA_REDACAO', 'Redação')
]

def gerar_boxplot_estatistico(df, coluna, titulo, cor_pr, cor_br):
    """Gera um boxplot agrupado usando estatísticas pré-calculadas"""
    fig = go.Figure()
    
    for regiao, label, cor in [('PR', 'Paraná (PR)', cor_pr), ('BR', 'Brasil (Sem PR)', cor_br)]:
        # Filtra os dados da região
        if regiao == 'PR':
            data = df[df['SG_UF_PROVA'] == 'PR'][coluna].dropna()
        else:
            data = df[df['SG_UF_PROVA'] != 'PR'][coluna].dropna()
            
        if not data.empty:
            # Cálculo dos quartis no servidor
            q1 = np.percentile(data, 25)
            median = np.percentile(data, 50)
            q3 = np.percentile(data, 75)
            low = data.min()
            high = data.max()
            
            fig.add_trace(go.Box(
                name=label,
                y=['Notas'], # <-- EIXO Y FIXO: Coloca ambos na mesma "linha" base
                q1=[q1], median=[median], q3=[q3],
                lowerfence=[low], upperfence=[high],
                fillcolor=cor, line=dict(color='white', width=1),
                orientation='h', marker_color=cor
            ))
            
    fig.update_layout(
        title=titulo,
        xaxis_title="Nota",
        yaxis=dict(visible=False), # Esconde a palavra 'Notas' para o visual ficar limpo
        boxmode='group',           # <-- O SEGREDO: Força o Plotly a colocar um ao lado/cima do outro (estilo hue)
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,           # Ativamos a legenda para identificar as cores
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

for i, (col, nome) in enumerate(colunas_nomes):
    with tabs[i]:
        c_esq, c_dir = st.columns(2)
        
        with c_esq:
            st.write(f"**{nome} (Antes)**")
            fig_a = obter_grafico_cache(f"box_{col}_antes.json", 
                                        lambda c=col: gerar_boxplot_estatistico(df_bruto, c, "Original", '#1f77b4', '#ff7f0e'))
            st.plotly_chart(fig_a, use_container_width=True)
            
        with c_dir:
            st.write(f"**{nome} (Depois)**")
            fig_d = obter_grafico_cache(f"box_{col}_depois.json", 
                                        lambda c=col: gerar_boxplot_estatistico(df_limpo, c, "Tratado", '#1f77b4', '#ff7f0e'))
            st.plotly_chart(fig_d, use_container_width=True)

st.markdown("---")

# =====================================================================
# 3. OTIMIZAÇÃO DE TIPOS E MEMÓRIA
# =====================================================================
st.header("3. Otimização Estrutural (Memory Downcast)")
st.write("A base de dados original consumia uma quantidade excessiva de memória RAM. Para viabilizar a execução fluida deste painel analítico na web, realizámos a alteração forçada dos tipos primitivos de dados.")

dados_tipos = {
    'Nome da Coluna': ['SG_UF_PROVA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'],
    'Tipo Original (.parquet bruto)': ['object', 'float64', 'float64', 'float64', 'float64', 'float64'],
    'Tipo Otimizado (.parquet limpo)': ['category', 'float32', 'float32', 'float32', 'float32', 'float32'],
    'Tamanho Novo (Por registo)': ['1 byte', '4 bytes', '4 bytes', '4 bytes', '4 bytes', '4 bytes']
}

st.dataframe(pd.DataFrame(dados_tipos), use_container_width=True, hide_index=True)