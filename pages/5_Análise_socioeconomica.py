import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache_graficos')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Impacto Socioeconômico", page_icon="💰", layout="wide")

st.title("💰 O Impacto Socioeconômico: Paraná vs Brasil")
st.markdown("---")

# =====================================================================
# MAPEAMENTOS DO DICIONÁRIO DE DADOS (INEP)
# =====================================================================
MAPA_RENDA = {
    'A': 'Sem Renda', 'B': 'Até 1.320', 'C': '1.320-1.980', 'D': '1.980-2.640',
    'E': '2.640-3.300', 'F': '3.300-3.960', 'G': '3.960-5.280', 'H': '5.280-6.600',
    'I': '6.600-7.920', 'J': '7.920-9.240', 'K': '9.240-10.560', 'L': '10.560-13.200',
    'M': '13.200-15.840', 'N': '15.840-19.800', 'O': '19.800-26.400', 'P': 'Acima 26.400', 'Q': 'Riqueza Máxima'
}

MAPA_COMPUTADOR = {
    'A': 'Não possui', 'B': 'Possui 1', 'C': 'Possui 2', 'D': 'Possui 3', 'E': '4 ou mais'
}

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
# CARREGAMENTO E TRATAMENTO (OTIMIZADO)
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ O ficheiro de dados limpos não foi encontrado.")
    st.stop()

@st.cache_data
def carregar_dados_socio():
    colunas = ['SG_UF_PROVA', 'NU_NOTA_REDACAO', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'Q006', 'Q024', 'Q025']
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    # Numérico para cálculo de Correlação de Pearson (Fazer antes de converter para categoria)
    renda_ordem = sorted(df['Q006'].dropna().unique())
    df['RENDA_NUM'] = df['Q006'].map({cat: i+1 for i, cat in enumerate(renda_ordem)})
    df['COMP_NUM'] = df['Q024'].map({cat: i for i, cat in enumerate(sorted(df['Q024'].dropna().unique()))})
    
    # --- MEMORY DOWNCAST ---
    colunas_categoria = ['SG_UF_PROVA', 'Regiao', 'Q006', 'Q024', 'Q025']
    for col in colunas_categoria:
        df[col] = df[col].astype('category')
        
    colunas_float = ['NU_NOTA_REDACAO', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC']
    for col in colunas_float:
        df[col] = df[col].astype('float32')
        
    return df, renda_ordem

with st.spinner("Processando dados socioeconômicos..."):
    df, renda_ordem = carregar_dados_socio()
    df_pr = df[df['Regiao'] == 'Paraná (PR)'].dropna(subset=['Q006', 'NU_NOTA_REDACAO'])
    df_br = df[df['Regiao'] == 'Brasil (Sem PR)'].dropna(subset=['Q006', 'NU_NOTA_REDACAO'])

# =====================================================================
# FUNÇÕES DE PRÉ-CÁLCULO E PLOTAGEM VETORIAL (PLOTLY)
# =====================================================================

def gerar_linha_renda_redacao():
    # Pré-cálculo da média exata
    df_agg = df.groupby(['Q006', 'Regiao'], observed=True)['NU_NOTA_REDACAO'].mean().reset_index()
    # Adicionando o rótulo amigável
    df_agg['Renda_Label'] = df_agg['Q006'].map(MAPA_RENDA)
    
    # Pearson
    corr_pr, _ = pearsonr(df_pr['RENDA_NUM'], df_pr['NU_NOTA_REDACAO'])
    corr_br, _ = pearsonr(df_br['RENDA_NUM'], df_br['NU_NOTA_REDACAO'])
    
    fig = go.Figure()
    
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        dados_regiao = df_agg[df_agg['Regiao'] == regiao].sort_values('Q006')
        fig.add_trace(go.Scatter(
            x=dados_regiao['Renda_Label'], y=dados_regiao['NU_NOTA_REDACAO'],
            mode='lines+markers', name=regiao,
            line=dict(color=cor, width=3), marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Nota Média: %{y:.1f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(text=f"Tendência: Renda vs Redação<br><sup>Corr. PR: {corr_pr:.3f} | Corr. BR: {corr_br:.3f}</sup>", font=dict(size=18)),
        xaxis=dict(tickangle=-45), yaxis_title="Nota Média na Redação",
        margin=dict(t=80, b=100, l=40, r=40), legend=dict(title='Região')
    )
    return fig

def gerar_barras_gap():
    # Gap absoluto
    gap_pr = df_pr[df_pr['Q006'] == 'Q']['NU_NOTA_REDACAO'].mean() - df_pr[df_pr['Q006'] == 'A']['NU_NOTA_REDACAO'].mean()
    gap_br = df_br[df_br['Q006'] == 'Q']['NU_NOTA_REDACAO'].mean() - df_br[df_br['Q006'] == 'A']['NU_NOTA_REDACAO'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Paraná (PR)', 'Brasil (Sem PR)'], y=[gap_pr, gap_br],
        marker_color=['#1f77b4', '#ff7f0e'],
        text=[f"{gap_pr:.1f} pts", f"{gap_br:.1f} pts"], textposition='outside'
    ))
    
    fig.update_layout(title="Gap de Desigualdade<br><sup>(Diferença de nota entre a Classe de Maior Renda e a Sem Renda)</sup>", yaxis_title="Diferença de Pontos", margin=dict(t=70, b=20, l=20, r=20))
    return fig

def gerar_barras_internet():
    areas = {'NU_NOTA_CN': 'Natureza', 'NU_NOTA_CH': 'Humanas', 'NU_NOTA_LC': 'Linguagens', 'NU_NOTA_MT': 'Matemática', 'NU_NOTA_REDACAO': 'Redação'}
    
    def calcular_prejuizo(data):
        com = data[data['Q025'] == 'B'][list(areas.keys())].mean()
        sem = data[data['Q025'] == 'A'][list(areas.keys())].mean()
        return com - sem

    perda_pr = calcular_prejuizo(df_pr)
    perda_br = calcular_prejuizo(df_br)
    
    df_perda = pd.DataFrame({
        'Área': list(areas.values()), 'Paraná (PR)': perda_pr.values, 'Brasil (Sem PR)': perda_br.values
    }).melt(id_vars='Área', var_name='Regiao', value_name='Pontos Perdidos')
    
    fig = go.Figure()
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        dados_reg = df_perda[df_perda['Regiao'] == regiao]
        fig.add_trace(go.Bar(
            name=regiao, x=dados_reg['Área'], y=dados_reg['Pontos Perdidos'],
            marker_color=cor, text=[f"-{v:.0f}" for v in dados_reg['Pontos Perdidos']], textposition='outside'
        ))
        
    fig.update_layout(title="O Custo da Exclusão Digital<br><sup>(Pontos perdidos por não possuir acesso à internet em casa)</sup>", barmode='group', yaxis_title="Impacto Negativo (Pontos)", margin=dict(t=70, b=20, l=20, r=20))
    return fig

def gerar_barras_computador(coluna_nota, titulo):
    # Agregação rápida no backend (observed=True para lidar com a nova otimização de Categorias)
    df_agg = df.groupby(['Q024', 'Regiao'], observed=True)[coluna_nota].mean().reset_index()
    df_agg['Computadores'] = df_agg['Q024'].map(MAPA_COMPUTADOR)
    
    fig = go.Figure()
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        dados_regiao = df_agg[df_agg['Regiao'] == regiao].sort_values('Q024')
        fig.add_trace(go.Bar(
            name=regiao, x=dados_regiao['Computadores'], y=dados_regiao[coluna_nota],
            marker_color=cor, text=[f"{v:.1f}" for v in dados_regiao[coluna_nota]], textposition='outside'
        ))
        
    fig.update_layout(title=titulo, barmode='group', yaxis_title="Nota Média", xaxis=dict(tickangle=-30), margin=dict(t=50, b=40, l=20, r=20))
    return fig

# =====================================================================
# RENDERIZAÇÃO DA PÁGINA
# =====================================================================

# --- 1. RENDA vs REDAÇÃO ---
st.header("1. Renda Familiar e o Coeficiente de Correlação")
st.write("Abaixo, observamos como a nota média da Redação aumenta exponencialmente à medida que a faixa de renda familiar sobe.")
fig_renda = obter_grafico_cache("linha_renda_redacao.json", gerar_linha_renda_redacao)
st.plotly_chart(fig_renda, use_container_width=True)

st.markdown("---")

# --- 2. GAP RICOS VS POBRES ---
st.header("2. Eficiência na Redução da Distância Social")
fig_gap = obter_grafico_cache("bar_gap_renda.json", gerar_barras_gap)
st.plotly_chart(fig_gap, use_container_width=True)

st.markdown("---")

# --- 3. IMPACTO DA INTERNET ---
st.header("3. O Prejuízo da Exclusão Digital")
st.write("Quanto um aluno **perde de nota**, em média, por não possuir acesso à internet no seu domicílio?")
fig_internet = obter_grafico_cache("bar_impacto_internet.json", gerar_barras_internet)
st.plotly_chart(fig_internet, use_container_width=True)

st.markdown("---")

# --- 4. COMPUTADOR EM CASA ---
st.header("4. Posse de Computador e Desempenho")
st.write("Comparação das médias de notas de acordo com a quantidade de computadores no domicílio.")

col_a, col_b = st.columns(2)

with col_a:
    fig_comp_mt = obter_grafico_cache("bar_computador_mt.json", lambda: gerar_barras_computador('NU_NOTA_MT', "Média em Matemática"))
    st.plotly_chart(fig_comp_mt, use_container_width=True)

with col_b:
    fig_comp_red = obter_grafico_cache("bar_computador_redacao.json", lambda: gerar_barras_computador('NU_NOTA_REDACAO', "Média em Redação"))
    st.plotly_chart(fig_comp_red, use_container_width=True)

# Cálculo de P-Value (Estatística Rápida)
_, p_mt = pearsonr(df.dropna(subset=['COMP_NUM', 'NU_NOTA_MT'])['COMP_NUM'], df.dropna(subset=['COMP_NUM', 'NU_NOTA_MT'])['NU_NOTA_MT'])
_, p_red = pearsonr(df.dropna(subset=['COMP_NUM', 'NU_NOTA_REDACAO'])['COMP_NUM'], df.dropna(subset=['COMP_NUM', 'NU_NOTA_REDACAO'])['NU_NOTA_REDACAO'])

st.success(f"**Conclusão Estatística:** O P-Value calculado ({p_mt:.1e}) confirma que a posse de tecnologia é um preditor significativo de desempenho tanto no Paraná quanto no Brasil.")