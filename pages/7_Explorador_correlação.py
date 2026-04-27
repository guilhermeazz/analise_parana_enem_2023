import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Explorador de Insights", page_icon="💡", layout="wide")

st.title("💡 Explorador de Correlações")
st.write("Utilize os filtros laterais e as seleções abaixo para descobrir padrões ocultos e correlações na base de dados do ENEM 2023.")
st.markdown("---")

# =====================================================================
# DICIONÁRIOS GLOBAIS
# =====================================================================
DICIONARIO_VARS = {
    'NU_NOTA_MT': 'Nota de Matemática',
    'NU_NOTA_REDACAO': 'Nota da Redação',
    'NU_NOTA_CN': 'Nota de C. da Natureza',
    'NU_NOTA_LC': 'Nota de Linguagens',
    'NU_NOTA_CH': 'Nota de C. Humanas',
    'Renda_Num': 'Faixa de Renda (Escala)',
    'Comp_Num': 'Qtd. Computadores (Escala)'
}

CATS_DISPONIVEIS = {
    'Q006': 'Renda Familiar', 
    'Escola_Label': 'Tipo de Escola', 
    'Regiao': 'Região do País', 
    'TP_SEXO': 'Sexo'
}

MAPA_RACA = {
    0: 'Não Declarado', 1: 'Branca', 2: 'Preta', 
    3: 'Parda', 4: 'Amarela', 5: 'Indígena'
}

# =====================================================================
# CARREGAMENTO OTIMIZADO DE DADOS
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ Arquivo de dados não encontrado.")
    st.stop()

@st.cache_data
def carregar_dados_dinamicos():
    """Carrega apenas o necessário e já mapeia variáveis para ordenação"""
    colunas = [
        'SG_UF_PROVA', 'TP_SEXO', 'TP_ESCOLA', 'Q006', 'Q024', 'TP_COR_RACA',
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'
    ]
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    # Mapeamento Numérico para Correlação (Transformando Categorias em Escala)
    renda_ordem = sorted(df['Q006'].dropna().unique())
    df['Renda_Num'] = df['Q006'].map({cat: i+1 for i, cat in enumerate(renda_ordem)})
    df['Comp_Num'] = df['Q024'].map({cat: i for i, cat in enumerate(sorted(df['Q024'].dropna().unique()))})
    
    # Mapeamentos Amigáveis
    df['Escola_Label'] = df['TP_ESCOLA'].map({1: 'Não Respondeu', 2: 'Pública', 3: 'Privada'})
    
    # --- MEMORY DOWNCAST ---
    colunas_categoria = ['SG_UF_PROVA', 'TP_SEXO', 'TP_ESCOLA', 'Q006', 'Q024', 'TP_COR_RACA', 'Regiao', 'Escola_Label']
    for col in colunas_categoria:
        df[col] = df[col].astype('category')
        
    colunas_float = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    for col in colunas_float:
        df[col] = df[col].astype('float32')
        
    return df

with st.spinner("A preparar o motor de dados..."):
    df_completo = carregar_dados_dinamicos()

# =====================================================================
# BARRA LATERAL (FILTROS GLOBAIS)
# =====================================================================
st.sidebar.header("Filtros Globais da Amostra")

filtro_regiao = st.sidebar.selectbox("Região de Análise:", ["Comparativo (Ambos)", "Apenas Paraná (PR)", "Apenas Brasil (Sem PR)"])
filtro_sexo = st.sidebar.radio("Sexo Biológico:", ["Todos", "Masculino (M)", "Feminino (F)"])

# Aplicação dos filtros rápidos via Máscara Booleana (Pandas Backend)
mask = pd.Series([True] * len(df_completo))

if filtro_regiao == "Apenas Paraná (PR)":
    mask &= (df_completo['Regiao'] == 'Paraná (PR)')
elif filtro_regiao == "Apenas Brasil (Sem PR)":
    mask &= (df_completo['Regiao'] == 'Brasil (Sem PR)')

if filtro_sexo == "Masculino (M)":
    mask &= (df_completo['TP_SEXO'] == 'M')
elif filtro_sexo == "Feminino (F)":
    mask &= (df_completo['TP_SEXO'] == 'F')

df_filtrado = df_completo[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Amostra Analisada", f"{len(df_filtrado):,}".replace(',', '.'))

if len(df_filtrado) == 0:
    st.warning("Nenhum dado encontrado para esta combinação de filtros.")
    st.stop()

# =====================================================================
# INSIGHT 1: MATRIZ DE CORRELAÇÃO 2D (DENSITY HEATMAP)
# =====================================================================
st.header("1. Motor de Correlação Variável")
st.write("Descubra a relação direta entre duas variáveis. O gráfico mostra a densidade de alunos: **áreas mais claras representam uma maior concentração de candidatos**.")

col_var1, col_var2 = st.columns(2)
with col_var1:
    var_x = st.selectbox("Selecione o Eixo X:", list(DICIONARIO_VARS.keys()), format_func=lambda x: DICIONARIO_VARS[x], index=0)
with col_var2:
    var_y = st.selectbox("Selecione o Eixo Y:", list(DICIONARIO_VARS.keys()), format_func=lambda x: DICIONARIO_VARS[x], index=1)

if var_x != var_y:
    df_corr = df_filtrado.dropna(subset=[var_x, var_y])
    
    if len(df_corr) > 10:
        coef, p_val = pearsonr(df_corr[var_x], df_corr[var_y])
        
        fig_corr = px.density_heatmap(
            df_corr, x=var_x, y=var_y, 
            color_continuous_scale="Viridis",
            title=f"Relação: {DICIONARIO_VARS[var_x]} vs {DICIONARIO_VARS[var_y]}"
        )
        
        fig_corr.add_annotation(
            text=f"Correlação de Pearson (R): <b>{coef:.2f}</b>",
            xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False,
            font=dict(size=14, color="white"), bgcolor="rgba(0,0,0,0.6)", borderpad=4
        )
        
        fig_corr.update_layout(
            xaxis_title=DICIONARIO_VARS[var_x], yaxis_title=DICIONARIO_VARS[var_y],
            margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        if coef > 0.5: st.success("Existe uma **forte relação positiva** entre estas variáveis.")
        elif coef > 0.2: st.info("Existe uma **relação positiva moderada/fraca** entre estas variáveis.")
        elif coef < -0.2: st.warning("Existe uma **relação negativa** (quando uma sobe, a outra desce).")
        else: st.write("Não há relação linear significativa entre estas variáveis.")
else:
    st.warning("Selecione variáveis diferentes para comparar.")

st.markdown("---")

# =====================================================================
# INSIGHT 2: MATRIZ DE CRUZAMENTO SOCIOECONÔMICO (PIVOT TABLE)
# =====================================================================
st.header("2. O Mapa de Desigualdades (Cruzamento Categórico)")
st.write("Cruze duas categorias demográficas para descobrir qual subgrupo atinge as maiores médias em uma prova específica.")

col_cat1, col_cat2, col_nota = st.columns(3)
with col_cat1:
    cat_y = st.selectbox("Linhas (Eixo Y):", list(CATS_DISPONIVEIS.keys()), format_func=lambda x: CATS_DISPONIVEIS[x], index=0)
with col_cat2:
    cat_x = st.selectbox("Colunas (Eixo X):", list(CATS_DISPONIVEIS.keys()), format_func=lambda x: CATS_DISPONIVEIS[x], index=1)
with col_nota:
    nota_target = st.selectbox("Analisar a Média de:", ['NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH'], format_func=lambda x: DICIONARIO_VARS[x])

if cat_x != cat_y:
    pivot_df = pd.pivot_table(
        df_filtrado, values=nota_target, index=cat_y, columns=cat_x, aggfunc='mean', observed=True
    ).round(1)
    
    if cat_y == 'Q006': pivot_df = pivot_df.loc[sorted(pivot_df.index)]
    if cat_x == 'Q006': pivot_df = pivot_df[sorted(pivot_df.columns)]
    
    fig_pivot = px.imshow(
        pivot_df, text_auto=".1f", aspect="auto", color_continuous_scale="RdBu_r",
        title=f"Média de {DICIONARIO_VARS[nota_target]} cruzando {CATS_DISPONIVEIS[cat_y]} e {CATS_DISPONIVEIS[cat_x]}"
    )
    
    fig_pivot.update_layout(
        xaxis_title=CATS_DISPONIVEIS[cat_x], yaxis_title=CATS_DISPONIVEIS[cat_y],
        margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pivot, use_container_width=True)
else:
    st.warning("Selecione categorias diferentes para cruzar os dados.")

st.markdown("---")

# =====================================================================
# INSIGHT 3: GRÁFICO DE CONTORNO DE DENSIDADE (TOPOGRÁFICO)
# =====================================================================
st.header("3. Mapa Topográfico de Desempenho (Density Contour)")
st.write("Este gráfico funciona como um mapa de relevo: as 'montanhas' (linhas mais fechadas) mostram exatamente onde se encontra o grande volume de alunos. É ideal para comparar a sobreposição entre o Paraná e o Brasil.")

col_c1, col_c2 = st.columns(2)
with col_c1:
    contorno_x = st.selectbox("Eixo X (Topografia):", ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'], format_func=lambda x: DICIONARIO_VARS[x], index=3)
with col_c2:
    contorno_y = st.selectbox("Eixo Y (Topografia):", ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'], format_func=lambda x: DICIONARIO_VARS[x], index=1)

if contorno_x != contorno_y:
    df_contorno = df_filtrado.dropna(subset=[contorno_x, contorno_y])
    
    amostra_pr = df_contorno[df_contorno['Regiao'] == 'Paraná (PR)'].sample(n=min(30000, len(df_contorno[df_contorno['Regiao'] == 'Paraná (PR)'])), random_state=42)
    amostra_br = df_contorno[df_contorno['Regiao'] == 'Brasil (Sem PR)'].sample(n=min(30000, len(df_contorno[df_contorno['Regiao'] == 'Brasil (Sem PR)'])), random_state=42)
    df_amostra_contorno = pd.concat([amostra_pr, amostra_br])

    fig_contour = px.density_contour(
        df_amostra_contorno, x=contorno_x, y=contorno_y, color="Regiao",
        color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'},
        title=f"Sobreposição de Concentração: {DICIONARIO_VARS[contorno_x]} vs {DICIONARIO_VARS[contorno_y]}"
    )
    
    fig_contour.update_traces(contours_coloring="none", line_width=2)
    fig_contour.update_layout(
        xaxis_title=DICIONARIO_VARS[contorno_x], yaxis_title=DICIONARIO_VARS[contorno_y],
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_contour, use_container_width=True)
else:
    st.warning("Selecione notas diferentes para gerar as curvas de nível.")

st.markdown("---")

# =====================================================================
# INSIGHT 4: DISPERSÃO COM DISTRIBUIÇÃO MARGINAL
# =====================================================================
st.header("4. Dispersão com Distribuição Marginal")
st.write("Cada ponto representa um aluno (usamos uma micro-amostra de 5.000 alunos para não travar o seu ecrã). Nos eixos externos, pode observar o histograma de distribuição das notas.")

df_scatter = df_filtrado.dropna(subset=['NU_NOTA_MT', 'NU_NOTA_REDACAO']).sample(n=min(5000, len(df_filtrado)), random_state=42)

fig_scatter = px.scatter(
    df_scatter, x="NU_NOTA_MT", y="NU_NOTA_REDACAO", color="Regiao",
    color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'},
    marginal_x="histogram", marginal_y="histogram",
    opacity=0.6,
    title="Correlação: Matemática vs Redação (Visão Micro-Amostral)"
)

fig_scatter.update_layout(
    xaxis_title="Nota de Matemática", yaxis_title="Nota da Redação",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")