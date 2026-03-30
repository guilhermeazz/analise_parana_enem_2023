import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Descrição Geral", page_icon="📖", layout="wide")

st.title("📖 Descrição Geral do Dataset")
st.markdown("---")

# =====================================================================
# CARREGAMENTO DOS DADOS
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ Dados não encontrados. Por favor, processe os dados na Home.")
    st.stop()

@st.cache_data
def carregar_resumo():
    colunas = ['SG_UF_PROVA', 'NU_NOTA_MT'] # MT apenas para contar registros válidos
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    return df

df = carregar_resumo()

# =====================================================================
# 1. ORIGEM E VOLUME DOS DADOS
# =====================================================================
st.header("1. Qual a origem e o volume dos dados analisados?")
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

# GRÁFICO DE PIZZA: PROPORÇÃO DA AMOSTRA
fig1, ax1 = plt.subplots(figsize=(6, 6))
labels = ['Paraná (PR)', 'Brasil (Sem PR)']
sizes = [total_pr, total_br]
colors = ['#1f77b4', '#ff7f0e']
explode = (0.1, 0)  # Destaca a fatia do Paraná

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=colors, textprops={'weight': 'bold'})
ax1.set_title("Proporção de Representatividade na Amostra", weight='bold')
st.pyplot(fig1)

st.markdown("---")

# =====================================================================
# 2. VARIÁVEIS UTILIZADAS
# =====================================================================
st.header("2. Quais variáveis compõem este estudo?")
st.write("""
Para realizar as análises estocásticas e socioeconômicas, selecionamos um conjunto estratégico de variáveis 
que permitem correlacionar o desempenho acadêmico com fatores externos.
""")

data_info = {
    "Categoria": ["Identificação", "Desempenho (Notas)", "Socioeconômico", "Socioeconômico", "Perfil"],
    "Variável Original": ["SG_UF_PROVA", "NU_NOTA_(CN, CH, LC, MT, REDACAO)", "Q006", "Q024 / Q025", "TP_SEXO / TP_FAIXA_ETARIA"],
    "Descrição": ["Estado de realização da prova", "Notas nas 5 áreas de conhecimento", "Renda mensal da família", "Posse de computador e internet", "Sexo biológico e idade agrupada"]
}
st.table(pd.DataFrame(data_info))

# GRÁFICO DE BARRAS: VOLUME DE DADOS POR ÁREA (COMPARAÇÃO)
st.write("**Disponibilidade de registros por Região**")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.barh(['Brasil (Sem PR)', 'Paraná (PR)'], [total_br, total_pr], color=['#ff7f0e', '#1f77b4'], alpha=0.8)
ax2.set_title("Volume Absoluto de Candidatos (Escala Logarítmica para Visualização)")
ax2.set_xscale('log') # Escala logarítmica para conseguir ver a barra do PR perto da do Brasil
ax2.set_xlabel("Quantidade de Inscritos (Log)")
ax2.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig2)

st.markdown("---")

# =====================================================================
# 3. CONTEXTO DO PROJETO
# =====================================================================
st.header("3. Como os dados foram preparados?")
st.write("""
Diferente da base bruta, os dados apresentados aqui passaram por um processo de **Engenharia de Dados**:
1. **Filtragem Geográfica:** Separação entre o público paranaense e o restante do país.
2. **Trimming Estocástico:** Remoção de outliers extremos para garantir que a média não fosse distorcida.
3. **Conversão de Tipos:** Otimização para garantir que o painel web funcione de forma rápida.
""")

# GRÁFICO DE ÁREA: REPRESENTAÇÃO VISUAL DA DENSIDADE
st.write("**Densidade de Registros (Exemplo: Matemática)**")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.kdeplot(df[df['Regiao'] == 'Paraná (PR)']['NU_NOTA_MT'], fill=True, color="#1f77b4", label="Paraná", ax=ax3)
sns.kdeplot(df[df['Regiao'] == 'Brasil (Sem PR)']['NU_NOTA_MT'], fill=True, color="#ff7f0e", label="Brasil", ax=ax3)
ax3.set_title("Distribuição Global das Notas na Amostra")
ax3.legend()
st.pyplot(fig3)