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
ARQUIVO_DADOS = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Perfil do Candidato", page_icon="👥", layout="wide")

st.title("👥 Perfil Sociodemográfico Comparativo: Paraná vs Brasil")
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
# VERIFICAÇÃO E CARREGAMENTO DOS DADOS (OTIMIZADO)
# =====================================================================
if not os.path.exists(ARQUIVO_DADOS):
    st.error(f"⚠️ O ficheiro de dados não foi encontrado na pasta 'data'.")
    st.stop()

@st.cache_data
def carregar_dados_perfil():
    """Lê apenas as colunas sociodemográficas essenciais para evitar sobrecarga na RAM"""
    colunas_perfil = [
        'SG_UF_PROVA', 'TP_SEXO', 'TP_COR_RACA', 
        'TP_FAIXA_ETARIA', 'IN_TREINEIRO', 'TP_ESCOLA'
    ]
    df_perfil = pd.read_parquet(ARQUIVO_DADOS, columns=colunas_perfil)
    
    df_perfil['TP_SEXO'] = df_perfil['TP_SEXO'].astype('category')
    df_perfil['TP_ESCOLA'] = pd.to_numeric(df_perfil['TP_ESCOLA'], errors='coerce')
    df_perfil['Regiao'] = np.where(df_perfil['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    return df_perfil

with st.spinner("A carregar e processar o perfil sociodemográfico..."):
    df = carregar_dados_perfil()

# =====================================================================
# FUNÇÕES DE PRÉ-CÁLCULO E PLOTAGEM VETORIAL (PLOTLY)
# =====================================================================
def gerar_barras_agrupadas(df_subset, coluna, mapa, titulo, rotacao_x=0):
    """Calcula a proporção em backend e plota barras interativas."""
    df_temp = df_subset.copy()
    df_temp[coluna] = df_temp[coluna].map(mapa)
    
    # Pre-calcula a proporção (100% da base)
    ct = pd.crosstab(df_temp[coluna], df_temp['Regiao'], normalize='columns') * 100
    
    fig = go.Figure()
    
    # Barra Paraná (Azul)
    if 'Paraná (PR)' in ct.columns:
        fig.add_trace(go.Bar(
            name='Paraná (PR)', x=ct.index, y=ct['Paraná (PR)'],
            marker_color='#1f77b4', text=[f"{v:.1f}%" for v in ct['Paraná (PR)']], textposition='outside'
        ))
        
    # Barra Brasil (Laranja)
    if 'Brasil (Sem PR)' in ct.columns:
        fig.add_trace(go.Bar(
            name='Brasil (Sem PR)', x=ct.index, y=ct['Brasil (Sem PR)'],
            marker_color='#ff7f0e', text=[f"{v:.1f}%" for v in ct['Brasil (Sem PR)']], textposition='outside'
        ))
        
    fig.update_layout(
        title=titulo, barmode='group',
        yaxis_title='Percentual dentro da Região (%)',
        xaxis_tickangle=rotacao_x, margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(title='Região', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def gerar_piramide_etaria(df_subset, titulo_regiao, cor_masc, cor_fem):
    """Calcula a distribuição etária e cria uma pirâmide horizontal (barmode='relative')"""
    labels_idade = [
        '< 17', '17', '18', '19', '20', '21', '22', '23', '24', '25',
        '26 a 30', '31 a 35', '36 a 40', '41 a 45', '46 a 50', 
        '51 a 55', '56 a 60', '61 a 65', '66 a 70', '> 70'
    ]
    mapa_idade = dict(enumerate(labels_idade, 1))
    
    df_temp = df_subset.dropna(subset=['TP_FAIXA_ETARIA', 'TP_SEXO']).copy()
    df_temp['Faixa Etária'] = df_temp['TP_FAIXA_ETARIA'].map(mapa_idade)
    
    tamanho_total = len(df_temp)
    if tamanho_total == 0: return go.Figure()
    
    # Cálculos exatos baseados na região
    df_m = (df_temp[df_temp['TP_SEXO'] == 'M']['Faixa Etária'].value_counts() / tamanho_total) * 100
    df_f = (df_temp[df_temp['TP_SEXO'] == 'F']['Faixa Etária'].value_counts() / tamanho_total) * 100
    
    # Extraindo na ordem correta
    m_vals = [-df_m.get(l, 0) for l in labels_idade] # Negativo para ir para a esquerda
    f_vals = [df_f.get(l, 0) for l in labels_idade]
    
    fig = go.Figure()
    
    # Masculino (Lado Esquerdo)
    fig.add_trace(go.Bar(
        y=labels_idade, x=m_vals, orientation='h', name='Masculino', 
        marker_color=cor_masc, text=[f"{abs(v):.1f}%" for v in m_vals], hoverinfo='text+name'
    ))
    
    # Feminino (Lado Direito)
    fig.add_trace(go.Bar(
        y=labels_idade, x=f_vals, orientation='h', name='Feminino', 
        marker_color=cor_fem, text=[f"{v:.1f}%" for v in f_vals], hoverinfo='text+name'
    ))
    
    # Lógica para os ticks do eixo X esconderem o número negativo
    max_val = max(max([abs(v) for v in m_vals]), max(f_vals))
    step = max(5, int(max_val / 5))
    tickvals = list(range(-int(max_val)-step, int(max_val)+step, step))
    ticktext = [f"{abs(t)}%" for t in tickvals]
    
    fig.update_layout(
        title=titulo_regiao, barmode='relative',
        xaxis=dict(tickvals=tickvals, ticktext=ticktext, title="Percentual da Região (%)"),
        yaxis=dict(title="Faixa Etária (Anos)"),
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# =====================================================================
# RENDERIZAÇÃO DAS PERGUNTAS DE ANÁLISE
# =====================================================================

# --- PERGUNTA 1: PERFIL DO CANDIDATO ---
st.header("1. Perfil sociodemográfico do candidato?")
st.write("Analisamos a distribuição de sexo, raça/cor e idade para entender as características populacionais dos inscritos, comparando o Paraná com a média Nacional.")

tab1, tab2 = st.tabs(["Distribuição por Sexo", "Distribuição por Raça/Cor"])

with tab1:
    st.write("**Distribuição Percentual por Sexo (Masculino vs Feminino)**")
    fig_sexo = obter_grafico_cache(
        "bar_sexo_comparativo.json", 
        lambda: gerar_barras_agrupadas(df, 'TP_SEXO', {'M': 'Masculino', 'F': 'Feminino'}, 'Perfil Comparativo: Sexo')
    )
    st.plotly_chart(fig_sexo, use_container_width=True)

with tab2:
    st.write("**Distribuição Percentual por Raça/Cor Declarada**")
    mapa_raca = {0: 'Não Declarado', 1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indígena'}
    fig_raca = obter_grafico_cache(
        "bar_raca_comparativo.json", 
        lambda: gerar_barras_agrupadas(df, 'TP_COR_RACA', mapa_raca, 'Perfil Comparativo: Raça/Cor', rotacao_x=-45)
    )
    st.plotly_chart(fig_raca, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# PIRÂMIDES ETÁRIAS 
st.write("**Pirâmide Etária Comparativa (Distribuição Etária por Sexo)**")
col_esq, col_dir = st.columns(2)

with col_esq:
    st.write("*Visualização do Paraná (Tons de Azul)*")
    fig_pir_pr = obter_grafico_cache(
        "piramide_parana.json",
        lambda: gerar_piramide_etaria(df[df['Regiao'] == 'Paraná (PR)'], 'Paraná (PR)', '#08519c', '#3182bd')
    )
    st.plotly_chart(fig_pir_pr, use_container_width=True)

with col_dir:
    st.write("*Visualização do Brasil (Tons de Laranja)*")
    fig_pir_br = obter_grafico_cache(
        "piramide_brasil.json",
        lambda: gerar_piramide_etaria(df[df['Regiao'] == 'Brasil (Sem PR)'], 'Brasil (Sem PR)', '#d94801', '#fd8d3c')
    )
    st.plotly_chart(fig_pir_br, use_container_width=True)

st.markdown("---")

# --- PERGUNTA 2: TREINEIROS ---
st.header("2. Taxa de treineiros?")
st.write("A taxa de treineiros mede a proporção de inscritos que ainda não concluíram o Ensino Médio e realizam a prova apenas para autoavaliação.")

fig_treineiro = obter_grafico_cache(
    "bar_treineiros_comparativo.json",
    lambda: gerar_barras_agrupadas(df, 'IN_TREINEIRO', {1: 'Treineiro', 0: 'Candidato Regular'}, 'Taxa Comparativa de Treineiros')
)
st.plotly_chart(fig_treineiro, use_container_width=True)

st.markdown("---")

# --- PERGUNTA 3: ESCOLAS PÚBLICAS VS PRIVADAS ---
st.header("3. Proporção de alunos de escolas públicas e privadas?")
st.write("Analisamos a origem escolar dos candidatos que declararam sua situação no momento da inscrição.")

fig_escola = obter_grafico_cache(
    "bar_escola_comparativo.json",
    lambda: gerar_barras_agrupadas(df[df['TP_ESCOLA'].isin([2, 3])], 'TP_ESCOLA', {2: 'Escola Pública', 3: 'Escola Privada'}, 'Proporção de Alunos: Pública vs Privada')
)
st.plotly_chart(fig_escola, use_container_width=True)