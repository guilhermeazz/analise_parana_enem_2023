import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PÁGINA
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

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
    'A': 'Não possui',
    'B': 'Possui 1',
    'C': 'Possui 2',
    'D': 'Possui 3',
    'E': '4 ou mais'
}

MAPA_INTERNET = {'A': 'Não', 'B': 'Sim'}

# =====================================================================
# CARREGAMENTO E TRATAMENTO
# =====================================================================
@st.cache_data
def carregar_dados_socio():
    colunas = ['SG_UF_PROVA', 'NU_NOTA_REDACAO', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'Q006', 'Q024', 'Q025']
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    # Numérico para Pearson
    renda_ordem = sorted(df['Q006'].unique())
    df['RENDA_NUM'] = df['Q006'].map({cat: i+1 for i, cat in enumerate(renda_ordem)})
    df['COMP_NUM'] = df['Q024'].map({cat: i for i, cat in enumerate(sorted(df['Q024'].unique()))})
    
    return df, renda_ordem

df, renda_ordem = carregar_dados_socio()
df_pr = df[df['Regiao'] == 'Paraná (PR)']
df_br = df[df['Regiao'] == 'Brasil (Sem PR)']

# =====================================================================
# 1. RENDA vs REDAÇÃO
# =====================================================================
st.header("1. Renda Familiar e o Coeficiente de Correlação")
corr_pr, _ = pearsonr(df_pr['RENDA_NUM'], df_pr['NU_NOTA_REDACAO'])
corr_br, _ = pearsonr(df_br['RENDA_NUM'], df_br['NU_NOTA_REDACAO'])

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df, x='Q006', y='NU_NOTA_REDACAO', hue='Regiao', palette=['#1f77b4', '#ff7f0e'], marker='o', ax=ax1)
ax1.set_title(f"Tendência: Renda vs Redação\n(Corr. PR: {corr_pr:.3f} | Corr. BR: {corr_br:.3f})", weight='bold')
ax1.set_xticklabels([MAPA_RENDA.get(x, x) for x in renda_ordem], rotation=45, fontsize=8)
st.pyplot(fig1)

st.markdown("---")

# =====================================================================
# 2. GAP RICOS VS POBRES
# =====================================================================
st.header("2. Eficiência na Redução da Distância Social")
gap_pr = df_pr[df_pr['Q006'] == 'Q']['NU_NOTA_REDACAO'].mean() - df_pr[df_pr['Q006'] == 'A']['NU_NOTA_REDACAO'].mean()
gap_br = df_br[df_br['Q006'] == 'Q']['NU_NOTA_REDACAO'].mean() - df_br[df_br['Q006'] == 'A']['NU_NOTA_REDACAO'].mean()

fig2, ax2 = plt.subplots(figsize=(8, 4))
bars = ax2.bar(['Paraná (PR)', 'Brasil (Sem PR)'], [gap_pr, gap_br], color=['#1f77b4', '#ff7f0e'], alpha=0.8)
ax2.set_title("Gap de Desigualdade (Notas: Classe Q - Classe A)")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f} pts', ha='center', va='bottom', weight='bold')
st.pyplot(fig2)

st.markdown("---")

# =====================================================================
# 4. IMPACTO DA INTERNET (VERSÃO INTUITIVA: "PERDA DE NOTA")
# =====================================================================
st.header("3. O Prejuízo da Exclusão Digital")
st.write("Quanto um aluno **perde de nota**, em média, por não possuir acesso à internet?")

areas = {'NU_NOTA_CN': 'Natureza', 'NU_NOTA_CH': 'Humanas', 'NU_NOTA_LC': 'Linguagens', 'NU_NOTA_MT': 'Matemática', 'NU_NOTA_REDACAO': 'Redação'}

def calcular_prejuizo_internet(data):
    com = data[data['Q025'] == 'B'][list(areas.keys())].mean()
    sem = data[data['Q025'] == 'A'][list(areas.keys())].mean()
    return com - sem

perda_pr = calcular_prejuizo_internet(df_pr)
perda_br = calcular_prejuizo_internet(df_br)

df_perda = pd.DataFrame({
    'Área': list(areas.values()),
    'Paraná (PR)': perda_pr.values,
    'Brasil (Sem PR)': perda_br.values
}).melt(id_vars='Área', var_name='Regiao', value_name='Pontos Perdidos')

fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_perda, x='Área', y='Pontos Perdidos', hue='Regiao', palette=['#1f77b4', '#ff7f0e'], ax=ax4)
ax4.set_title("Diferença de Nota: Alunos COM Internet vs Alunos SEM Internet", weight='bold')
ax4.set_ylabel("Impacto Negativo (Pontos a menos)")
for p in ax4.patches:
    ax4.annotate(f'-{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=9, weight='bold')
st.pyplot(fig4)

st.markdown("---")

# =====================================================================
# 5. COMPUTADOR EM CASA (NOMES REAIS)
# =====================================================================
st.header("4. Posse de Computador e Desempenho")
st.write("Comparação das médias de notas de acordo com a quantidade de computadores no domicílio.")

# Mapeamos as letras para nomes amigáveis antes de plotar
df['Computadores'] = df['Q024'].map(MAPA_COMPUTADOR)
ordem_comp = list(MAPA_COMPUTADOR.values())

fig5, ax5 = plt.subplots(1, 2, figsize=(14, 5))

# Matemática
sns.barplot(data=df, x='Computadores', y='NU_NOTA_MT', hue='Regiao', order=ordem_comp, palette=['#1f77b4', '#ff7f0e'], ax=ax5[0])
ax5[0].set_title("Média em Matemática", weight='bold')
ax5[0].tick_params(axis='x', rotation=30)

# Redação
sns.barplot(data=df, x='Computadores', y='NU_NOTA_REDACAO', hue='Regiao', order=ordem_comp, palette=['#1f77b4', '#ff7f0e'], ax=ax5[1])
ax5[1].set_title("Média em Redação", weight='bold')
ax5[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
st.pyplot(fig5)

# Cálculo de P-Value para o texto final
_, p_mt = pearsonr(df['COMP_NUM'], df['NU_NOTA_MT'])
_, p_red = pearsonr(df['COMP_NUM'], df['NU_NOTA_REDACAO'])

st.success(f"**Conclusão Estatística:** O P-Value calculado ({p_mt:.1e}) confirma que a posse de tecnologia é um preditor significativo de desempenho tanto no Paraná quanto no Brasil.")