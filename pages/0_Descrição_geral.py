import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache_graficos')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Descrição Geral", page_icon="📖", layout="wide")

st.title("📖 Descrição Geral do Dataset")
st.markdown("---")

# =====================================================================
# FUNÇÃO DE MOTOR DE CACHE (JSON)
# =====================================================================
def obter_grafico_cache(nome_arquivo, funcao_geradora):
    caminho = os.path.join(CACHE_DIR, nome_arquivo)
    if os.path.exists(caminho):
        return pio.read_json(caminho)
    
    fig = funcao_geradora()
    pio.write_json(fig, caminho)
    return fig

# =====================================================================
# CARREGAMENTO DOS DADOS (PARA MÉTRICAS)
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ Dados não encontrados. Por favor, processe os dados na Home.")
    st.stop()

@st.cache_data
def carregar_resumo_metricas():
    colunas = ['SG_UF_PROVA', 'NU_NOTA_MT']
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    return df

df = carregar_resumo_metricas()

# =====================================================================
# 1. ORIGEM E VOLUME DOS DADOS
# =====================================================================
st.header("1. Origem e o volume dos dados analisados")
st.write("""
Os dados foram extraídos dos **Microdados do ENEM 2023**, disponibilizados pelo **INEP** (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira). 
A base de dados original contém informações de milhões de inscritos, incluindo notas, perfis socioeconômicos e questionários contextuais.
""")

total_geral = len(df)
total_pr = len(df[df['Regiao'] == 'Paraná (PR)'])
total_br = len(df[df['Regiao'] == 'Brasil (Sem PR)'])

c1, c2, c3 = st.columns(3)
c1.metric("Total de Candidatos (Amostra Limpa)", f"{total_geral:,}".replace(',', '.'))
c2.metric("Candidatos no Paraná", f"{total_pr:,}".replace(',', '.'))
c3.metric("Candidatos no Brasil (Restante)", f"{total_br:,}".replace(',', '.'))

# Gráfico de Pizza (Restaurado com Plotly + Cache)
def gerar_pizza_proporcao():
    df_pie = pd.DataFrame({
        'Regiao': ['Paraná (PR)', 'Brasil (Sem PR)'],
        'Quantidade': [total_pr, total_br]
    })
    fig = px.pie(
        df_pie, values='Quantidade', names='Regiao',
        title="Proporção de Representatividade na Amostra",
        color='Regiao',
        color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'},
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
    return fig

fig1 = obter_grafico_cache("pie_representatividade.json", gerar_pizza_proporcao)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# =====================================================================
# 2. VARIÁVEIS UTILIZADAS
# =====================================================================
st.header("2. Variáveis compõem este estudo")
st.write("""
Para realizar as análises estocásticas e socioeconômicas, selecionamos um conjunto estratégico de variáveis 
que permitem correlacionar o desempenho acadêmico com fatores externos.
""")

data_info = {
    "Categoria": ["Identificação", "Desempenho (Notas)", "Socioeconômico", "Socioeconômico", "Perfil"],
    "Variável Original": ["SG_UF_PROVA", "NU_NOTA_(CN, CH, LC, MT, REDACAO)", "Q006", "Q024 / Q025", "TP_SEXO / TP_FAIXA_ETARIA"],
    "Descrição": ["Estado de realização da prova", "Notas nas 5 áreas de conhecimento", "Renda mensal da família", "Posse de computador e internet", "Sexo biológico e idade agrupada"]
}
st.dataframe(pd.DataFrame(data_info), use_container_width=True, hide_index=True)

# Gráfico de Barras Log (Restaurado com Plotly + Cache)
st.write("**Disponibilidade de registros por Região**")

def gerar_barras_volume():
    df_bar = pd.DataFrame({
        'Regiao': ['Brasil (Sem PR)', 'Paraná (PR)'],
        'Quantidade': [total_br, total_pr]
    })
    fig = px.bar(
        df_bar, x='Quantidade', y='Regiao', orientation='h',
        log_x=True,
        title="Volume Absoluto de Candidatos (Escala Logarítmica)",
        color='Regiao',
        color_discrete_map={'Paraná (PR)': '#1f77b4', 'Brasil (Sem PR)': '#ff7f0e'}
    )
    fig.update_layout(showlegend=False, xaxis_title="Quantidade de Inscritos (Log)", yaxis_title="")
    return fig

fig2 = obter_grafico_cache("bar_volume_log.json", gerar_barras_volume)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# =====================================================================
# 3. CONTEXTO DO PROJETO
# =====================================================================
st.header("3. Preparação dos dados")
st.write("""
Diferente da base bruta, os dados apresentados aqui passaram por um processo de **Engenharia de Dados**:
1. **Filtragem Geográfica:** Separação entre o público paranaense e o restante do país.
2. **Trimming Estocástico:** Remoção de outliers extremos para garantir que a média não fosse distorcida.
3. **Conversão de Tipos:** Otimização para garantir que o painel web funcione de forma rápida.
""")

# Gráfico de Densidade (Restaurado com Plotly + Cache)
st.write("**Densidade de Registros (Exemplo: Matemática)**")

def gerar_densidade_mt():
    # Amostragem para o gráfico de densidade não ficar pesado no JSON
    df_sample = df.sample(n=min(100000, len(df)), random_state=42)
    notas_pr = df_sample[df_sample['Regiao'] == 'Paraná (PR)']['NU_NOTA_MT'].dropna()
    notas_br = df_sample[df_sample['Regiao'] == 'Brasil (Sem PR)']['NU_NOTA_MT'].dropna()
    
    fig = ff.create_distplot(
        [notas_pr, notas_br], 
        group_labels=['Paraná', 'Brasil'], 
        show_hist=False,
        colors=['#1f77b4', '#ff7f0e']
    )
    fig.update_layout(
        title="Distribuição Global das Notas na Amostra (Matemática)",
        xaxis_title="Nota",
        yaxis_title="Densidade",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

fig3 = obter_grafico_cache("density_matematica.json", gerar_densidade_mt)
st.plotly_chart(fig3, use_container_width=True)