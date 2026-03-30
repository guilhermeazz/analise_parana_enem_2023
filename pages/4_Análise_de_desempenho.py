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

st.set_page_config(page_title="Análise de Desempenho", page_icon="📈", layout="wide")

st.title("📈 Análise de Desempenho: Áreas, Línguas e Redação")
st.markdown("---")

# =====================================================================
# CARREGAMENTO DOS DADOS
# =====================================================================
if not os.path.exists(ARQUIVO_LIMPO):
    st.error("⚠️ Dados limpos não encontrados. Por favor, processe os dados na Home.")
    st.stop()

@st.cache_data
def carregar_dados_desempenho():
    colunas = [
        'SG_UF_PROVA', 'TP_LINGUA',
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',
        'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5'
    ]
    df = pd.read_parquet(ARQUIVO_LIMPO, columns=colunas)
    df['Regiao'] = np.where(df['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    return df

df = carregar_dados_desempenho()
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
nomes_areas = ['C. Natureza', 'C. Humanas', 'Linguagens', 'Matemática', 'Redação']

# =====================================================================
# 1. ÁREAS DE MAIOR DESEMPENHO (MÉDIAS)
# =====================================================================
st.header("1. Quais as áreas de maior desempenho?")
st.write("Comparação das médias aritméticas das notas nas 5 áreas do conhecimento.")

df_medias = df.groupby('Regiao')[colunas_notas].mean().T
df_medias.index = nomes_areas
df_medias = df_medias[['Paraná (PR)', 'Brasil (Sem PR)']]

fig1, ax1 = plt.subplots(figsize=(10, 5))
df_medias.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'], alpha=0.9, width=0.7)
ax1.set_title('Média de Notas por Área do Conhecimento', weight='bold')
ax1.set_ylabel('Nota Média')
ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=0)

for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', weight='bold', fontsize=9)

st.pyplot(fig1)

st.markdown("---")

# =====================================================================
# 2. DISPERSÃO (DESVIO PADRÃO)
# =====================================================================
st.header("2. Qual área possui maior dispersão?")
st.write("O Desvio Padrão indica o nível de desigualdade: quanto maior o valor, mais heterogêneo é o desempenho dos alunos.")

col_esc, col_dir = st.columns([1, 2])

with col_esc:
    df_std = df.groupby('Regiao')[colunas_notas].std().T
    df_std.index = nomes_areas
    df_std = df_std[['Paraná (PR)', 'Brasil (Sem PR)']]
    st.dataframe(df_std.style.format("{:.2f}"), use_container_width=True)

with col_dir:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df_std.plot(kind='bar', ax=ax2, color=['#1f77b4', '#ff7f0e'], alpha=0.9)
    ax2.set_title('Dispersão (Desvio Padrão) por Área', weight='bold')
    ax2.set_ylabel('Valor do Desvio Padrão')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=0)
    st.pyplot(fig2)

st.markdown("---")

# =====================================================================
# 3. LÍNGUA ESTRANGEIRA (INGLÊS VS ESPANHOL)
# =====================================================================
st.header("3. Escolha de Língua Estrangeira")
mapa_lingua = {0: 'Inglês', 1: 'Espanhol'}

col_a, col_b = st.columns(2)

with col_a:
    st.write("**Proporção de Escolha**")
    ct_lingua = pd.crosstab(df['Regiao'], df['TP_LINGUA'].map(mapa_lingua), normalize='index') * 100
    ct_lingua = ct_lingua[['Inglês', 'Espanhol']]
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ct_lingua.T.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e'], width=0.7)
    ax3.set_title('Proporção de Escolha: Inglês vs Espanhol', weight='bold')
    ax3.set_ylabel('Percentual (%)')
    ax3.tick_params(axis='x', rotation=0)
    st.pyplot(fig3)

with col_b:
    st.write("**Desempenho em Linguagens por Opção de Língua**")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    df_lingua_plot = df.copy()
    df_lingua_plot['Língua'] = df_lingua_plot['TP_LINGUA'].map(mapa_lingua)
    
    sns.boxplot(data=df_lingua_plot, x='Língua', y='NU_NOTA_LC', hue='Regiao', 
                palette=['#1f77b4', '#ff7f0e'], ax=ax4)
    ax4.set_title('Boxplot: Notas de Linguagens (LC)', weight='bold')
    ax4.set_ylabel('Nota')
    st.pyplot(fig4)

st.markdown("---")

# =====================================================================
# 4. COMPETÊNCIAS DA REDAÇÃO (TABELA E RADAR)
# =====================================================================
st.header("4. Competências da Redação")
st.write("Análise das 5 competências avaliadas na Redação (0 a 200 pontos cada).")

col_comp = ['NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
labels_comp = ['Gramática', 'Tema/Repertório', 'Argumentação', 'Coesão', 'Proposta Interv.']

df_comp = df.groupby('Regiao')[col_comp].mean().T
df_comp.index = labels_comp
df_comp = df_comp[['Paraná (PR)', 'Brasil (Sem PR)']]

st.table(df_comp.style.format("{:.2f}"))

# GRÁFICO DE RADAR
def plotar_radar(df_radar):
    # 1. Preparação dos dados
    categorias = list(df_radar.index)
    N = len(categorias)
    
    # Ângulos para cada eixo (dividindo os 360 graus em 5 partes)
    angulos = [n / float(N) * 2 * np.pi for n in range(N)]
    angulos += angulos[:1] # fecha o polígono
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # -----------------------------------------------------------------
    # O PULO DO GATO: Ajuste de escala para dar "Zoom" nas diferenças
    # -----------------------------------------------------------------
    # Descobrimos a menor e maior nota para definir os limites do radar
    min_nota = df_radar.min().min()
    max_nota = df_radar.max().max()
    
    # Definimos o limite inferior um pouco abaixo da menor nota (ex: 15 pontos abaixo)
    # Isso faz com que as diferenças de 2 ou 3 pontos pareçam gigantes
    limite_inferior = max(0, min_nota - 15) 
    limite_superior = 160 # A nota máxima da competência é sempre 200
    
    ax.set_ylim(limite_inferior, limite_superior)
    # -----------------------------------------------------------------

    # Plotar Paraná (PR) - Azul
    valores_pr = df_radar['Paraná (PR)'].tolist()
    valores_pr += valores_pr[:1]
    ax.plot(angulos, valores_pr, linewidth=3, linestyle='solid', label='Paraná (PR)', color='#1f77b4', marker='o')
    ax.fill(angulos, valores_pr, '#1f77b4', alpha=0.2)
    
    # Plotar Brasil (BR) - Laranja
    valores_br = df_radar['Brasil (Sem PR)'].tolist()
    valores_br += valores_br[:1]
    ax.plot(angulos, valores_br, linewidth=3, linestyle='dashdot', label='Brasil (Sem PR)', color='#ff7f0e', marker='s')
    ax.fill(angulos, valores_br, '#ff7f0e', alpha=0.1)
    
    # Ajustes das legendas e eixos
    plt.xticks(angulos[:-1], categorias, color='black', size=11, weight='bold')
    
    # Customização dos círculos de fundo (Grids)
    ticks_calculados = np.linspace(limite_inferior, limite_superior, 5)
    ax.set_rticks(ticks_calculados)
    ax.set_yticklabels([f"{int(t)}" for t in ticks_calculados], color="grey", size=9)
    
    ax.set_rlabel_position(30) # Rotaciona os números do eixo para não bater na linha
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, shadow=True)
    ax.set_title("Destaques e Dificuldades: Radar de Competências\n(Escala Ajustada para Comparação)", 
                 weight='bold', size=14, pad=20)
    
    return fig

st.pyplot(plotar_radar(df_comp))