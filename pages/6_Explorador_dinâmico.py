import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Explorador Avançado", page_icon="🎛️", layout="wide")

st.title("🎛️ Explorador Dinâmico")
st.write("Utilize a barra lateral para segmentar os dados. Os gráficos abaixo recalcularão as médias e as distribuições de notas em tempo real.")
st.markdown("---")

# =====================================================================
# DICIONÁRIOS E MAPEAMENTOS
# =====================================================================
DICIONARIO_NOTAS = {
    'NU_NOTA_MT': 'Matemática',
    'NU_NOTA_REDACAO': 'Redação',
    'NU_NOTA_CN': 'Ciências da Natureza',
    'NU_NOTA_LC': 'Linguagens',
    'NU_NOTA_CH': 'Ciências Humanas'
}

# =====================================================================
# CARREGAMENTO DE DADOS (CACHE)
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ Ficheiro de dados não encontrado.")
    st.stop()

@st.cache_data
def carregar_dados_interativos():
    colunas = [
        'SG_UF_PROVA', 'TP_SEXO', 'TP_ESCOLA', 'Q006', 'Q025', 'TP_COR_RACA',
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'
    ]
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    # Aplicar rótulos amigáveis para os gráficos
    df['Escola'] = df['TP_ESCOLA'].map({1: 'Não Respondeu', 2: 'Pública', 3: 'Privada'})
    df['Internet'] = df['Q025'].map({'A': 'Sem Internet', 'B': 'Com Internet'})
    df['Sexo'] = df['TP_SEXO'].map({'M': 'Masculino', 'F': 'Feminino'})
    
    mapa_raca = {0: 'Não Decl.', 1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indígena'}
    df['Cor/Raça'] = df['TP_COR_RACA'].map(mapa_raca)
    
    # Agrupamento de Renda Simplificado
    def agrupar_renda(x):
        if x in ['A','B','C']: return 'Baixa (Até 2k)'
        elif x in ['D','E','F','G']: return 'Média-Baixa (Até 5k)'
        elif x in ['H','I','J','K']: return 'Média-Alta (Até 10k)'
        elif pd.notna(x): return 'Alta (Mais de 10k)'
        return 'Não Informado'
    df['Faixa de Renda'] = df['Q006'].apply(agrupar_renda)
    
    return df

with st.spinner("A preparar o motor analítico..."):
    df_completo = carregar_dados_interativos()

# =====================================================================
# BARRA LATERAL (MÚLTIPLOS FILTROS)
# =====================================================================
st.sidebar.header("🎯 Filtros Dinâmicos")

# Filtros Multiselect (Permitem escolher várias opções ao mesmo tempo)
filtro_sexo = st.sidebar.multiselect("Sexo:", df_completo['Sexo'].dropna().unique(), default=df_completo['Sexo'].dropna().unique())
filtro_escola = st.sidebar.multiselect("Tipo de Escola:", ['Pública', 'Privada'], default=['Pública', 'Privada'])
filtro_raca = st.sidebar.multiselect("Cor/Raça:", df_completo['Cor/Raça'].dropna().unique(), default=df_completo['Cor/Raça'].dropna().unique())
filtro_internet = st.sidebar.multiselect("Acesso à Internet:", df_completo['Internet'].dropna().unique(), default=df_completo['Internet'].dropna().unique())

# Aplicação dos Filtros (Lógica Booleana Pandas)
mask = (
    df_completo['Sexo'].isin(filtro_sexo) &
    df_completo['Escola'].isin(filtro_escola) &
    df_completo['Cor/Raça'].isin(filtro_raca) &
    df_completo['Internet'].isin(filtro_internet)
)
df_filtrado = df_completo[mask]

# Métricas da Barra Lateral
st.sidebar.markdown("---")
st.sidebar.metric("Tamanho da Amostra Filtrada", f"{len(df_filtrado):,}".replace(',', '.'))
if len(df_completo) > 0:
    perc_amostra = (len(df_filtrado) / len(df_completo)) * 100
    st.sidebar.write(f"*(Representa {perc_amostra:.1f}% da base total)*")

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhum candidato encontrado com a combinação de filtros atual. Altere as opções na barra lateral.")
    st.stop()

# =====================================================================
# CONTROLES DO GRÁFICO (EIXO X E NOTA)
# =====================================================================
st.write("### Configure a Análise Visual")
c1, c2 = st.columns(2)

with c1:
    nota_selecionada = st.selectbox("Qual Nota deseja analisar (Eixo Y)?", list(DICIONARIO_NOTAS.keys()), format_func=lambda x: DICIONARIO_NOTAS[x], index=0)
with c2:
    eixo_x_selecionado = st.selectbox("Dividir os dados por (Eixo X):", ['Faixa de Renda', 'Cor/Raça', 'Escola', 'Internet', 'Sexo'], index=0)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# GRÁFICO 1: BARRAS AGRUPADAS (MÉDIA EXATA)
# =====================================================================
st.subheader(f"📊 Média de {DICIONARIO_NOTAS[nota_selecionada]} por {eixo_x_selecionado}")

# Calcula a média exata usando 100% dos dados filtrados
df_barras = df_filtrado.groupby([eixo_x_selecionado, 'Regiao'])[nota_selecionada].mean().reset_index()

# Ordenação inteligente se for Renda
if eixo_x_selecionado == 'Faixa de Renda':
    ordem_renda = ['Baixa (Até 2k)', 'Média-Baixa (Até 5k)', 'Média-Alta (Até 10k)', 'Alta (Mais de 10k)']
    df_barras[eixo_x_selecionado] = pd.Categorical(df_barras[eixo_x_selecionado], categories=ordem_renda, ordered=True)
    df_barras = df_barras.sort_values(eixo_x_selecionado)

fig_bar = px.bar(
    df_barras, 
    x=eixo_x_selecionado, 
    y=nota_selecionada, 
    color='Regiao',
    barmode='group',
    color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'},
    text_auto='.1f',
    title=f"Comparativo de Médias: PR vs BR (Amostra Filtrada)"
)

fig_bar.update_traces(textposition='outside')
fig_bar.update_layout(
    yaxis_title=f"Nota Média ({DICIONARIO_NOTAS[nota_selecionada]})",
    xaxis_title=eixo_x_selecionado,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=50, b=20, l=20, r=20),
    legend_title="Região"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# =====================================================================
# GRÁFICO 2: BOXPLOT DINÂMICO (DISTRIBUIÇÃO)
# =====================================================================
st.subheader(f"📦 Distribuição e Outliers (Boxplot agrupado por {eixo_x_selecionado})")
st.write("O Boxplot mostra a variação real das notas (Mediana, Quartis e Extremos). Para garantir a fluidez do painel, os gráficos abaixo utilizam uma amostra perfeitamente representativa dos dados filtrados.")

# AMOSTRAGEM INTELIGENTE: Pega no máximo 40.000 linhas da base filtrada para não travar o Plotly
df_boxplot = df_filtrado.dropna(subset=[nota_selecionada, eixo_x_selecionado])
if len(df_boxplot) > 40000:
    df_boxplot = df_boxplot.sample(n=40000, random_state=42)

# Ordenação para o Boxplot
if eixo_x_selecionado == 'Faixa de Renda':
    df_boxplot[eixo_x_selecionado] = pd.Categorical(df_boxplot[eixo_x_selecionado], categories=ordem_renda, ordered=True)
    df_boxplot = df_boxplot.sort_values(eixo_x_selecionado)

# px.box gera automaticamente os boxplots lado a lado quando usamos a cor
fig_box = px.box(
    df_boxplot, 
    x=eixo_x_selecionado, 
    y=nota_selecionada, 
    color='Regiao',
    color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'},
    title=f"Dispersão de Notas: {DICIONARIO_NOTAS[nota_selecionada]}",
)

# O segredo para os Boxplots não ficarem uns em cima dos outros no Plotly Express
fig_box.update_layout(
    boxmode='group',
    yaxis_title=f"Nota ({DICIONARIO_NOTAS[nota_selecionada]})",
    xaxis_title=eixo_x_selecionado,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=50, b=20, l=20, r=20)
)

# Transparência no preenchimento do Boxplot
fig_box.update_traces(marker=dict(opacity=0.7), line=dict(width=2))

st.plotly_chart(fig_box, use_container_width=True)

# =====================================================================
# MÉTRICAS FINAIS COMPARATIVAS
# =====================================================================
st.markdown("---")
st.write("### 🧮 Resumo Estatístico da Amostra Atual")

# Calcula os números apenas com os dados que passaram pelo filtro
media_pr = df_filtrado[df_filtrado['Regiao'] == 'Paraná (PR)'][nota_selecionada].mean()
media_br = df_filtrado[df_filtrado['Regiao'] == 'Brasil (Sem PR)'][nota_selecionada].mean()

if pd.notna(media_pr) and pd.notna(media_br):
    diferenca = media_pr - media_br
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(f"Média Paraná ({DICIONARIO_NOTAS[nota_selecionada]})", f"{media_pr:.1f}")
    col_m2.metric(f"Média Brasil ({DICIONARIO_NOTAS[nota_selecionada]})", f"{media_br:.1f}")
    col_m3.metric("Vantagem do Paraná (PR - BR)", f"{diferenca:+.1f} pts", delta_color="normal")
else:
    st.info("Não há dados suficientes nas duas regiões para gerar as métricas de diferença.")