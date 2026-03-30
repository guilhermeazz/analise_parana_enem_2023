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

ARQUIVO_BRUTO = os.path.join(DATA_DIR, 'enem2023.parquet')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Limpeza de Dados", page_icon="🧹", layout="wide")

st.title("🧹 Processo de Limpeza e Preparação dos Dados")
st.markdown("---")

# =====================================================================
# CARREGAMENTO DOS DADOS (COM CACHE PARA PERFORMANCE)
# =====================================================================
@st.cache_data
def carregar_dados_brutos():
    """Carrega apenas as colunas essenciais da base bruta original"""
    colunas_alvo = ['SG_UF_PROVA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    return pd.read_parquet(ARQUIVO_BRUTO, columns=colunas_alvo)

@st.cache_data
def carregar_dados_limpos():
    """Carrega a base final já tratada (sem outliers e sem nulos)"""
    return pd.read_parquet(ARQUIVO_LIMPO)

# Carregando as duas bases em memória
df_bruto = carregar_dados_brutos()
df_limpo = carregar_dados_limpos()

# =====================================================================
# SEÇÃO 1: DADOS FALTANTES (CÁLCULO DINÂMICO E CONSISTENTE)
# =====================================================================
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

# Separa a base bruta por região para contar as faltas reais
df_pr_bruto = df_bruto[df_bruto['SG_UF_PROVA'] == 'PR']
df_br_bruto = df_bruto[df_bruto['SG_UF_PROVA'] != 'PR']

# Conta quantas linhas possuem pelo menos um dado faltante
faltantes_pr = int(df_pr_bruto[colunas_notas].isnull().any(axis=1).sum())
faltantes_br = int(df_br_bruto[colunas_notas].isnull().any(axis=1).sum())

st.header("1. Tratamento de Dados Faltantes")
st.write("A base de dados bruta do ENEM possui um elevado índice de abstenções, o que gera dados nulos (NaN) nas colunas de notas. Para garantir a integridade da modelagem estatística, todos os alunos com notas em branco foram removidos da nossa amostra.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tamanho Original da Base", f"{len(df_bruto):,}".replace(',', '.'))
col2.metric("Tamanho Após Limpeza", f"{len(df_limpo):,}".replace(',', '.'))
col3.metric("Faltantes (Brasil sem PR)", f"{faltantes_br:,}".replace(',', '.'))
col4.metric("Faltantes (Paraná)", f"{faltantes_pr:,}".replace(',', '.'))

st.markdown("---")

# =====================================================================
# SEÇÃO 2: OUTLIERS (COMPARAÇÃO ANTES E DEPOIS)
# =====================================================================
st.header("2. Tratamento de Outliers (Boxplots)")
st.write("""
Para evitar distorções nos modelos preditivos e garantir a estabilização da variância estocástica, optámos pela técnica de **Truncamento (Trimming)** baseada no Intervalo Interquartil (IQR). 

Abaixo é possível visualizar a distribuição dos dados de forma comparativa: do lado esquerdo, a base original contendo as anomalias e *outliers* extremos; do lado direito, o resultado estatisticamente robusto após a aplicação do filtro IQR.
""")

tabs = st.tabs(["Ciências da Natureza", "Ciências Humanas", "Linguagens", "Matemática", "Redação"])

colunas_nomes = [
    ('NU_NOTA_CN', 'Ciências da Natureza'),
    ('NU_NOTA_CH', 'Ciências Humanas'),
    ('NU_NOTA_LC', 'Linguagens e Códigos'),
    ('NU_NOTA_MT', 'Matemática'),
    ('NU_NOTA_REDACAO', 'Redação')
]

def plotar_boxplot_comparativo(df_plot, coluna, titulo, subtitulo, cor_fundo):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Prepara um DataFrame temporário (Remove NaNs apenas para o Seaborn não falhar no plot bruto)
    df_temp = df_plot[['SG_UF_PROVA', coluna]].dropna().copy()
    df_temp['Regiao_Label'] = np.where(df_temp['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    sns.boxplot(
        data=df_temp, 
        x=coluna, 
        y='Regiao_Label', 
        ax=ax, 
        palette=['#1f77b4', '#ff7f0e'], 
        order=['Paraná (PR)', 'Brasil (Sem PR)']
    )
    
    ax.set_title(f'{titulo}\n{subtitulo}', weight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('Nota Avaliada')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adiciona um leve tom de fundo ao gráfico para diferenciar visualmente o Antes e Depois
    ax.set_facecolor(cor_fundo)
    
    return fig

# Desenha os gráficos lado a lado dentro de cada tab
for i, (col, nome) in enumerate(colunas_nomes):
    with tabs[i]:
        col_esquerda, col_direita = st.columns(2)
        
        with col_esquerda:
            # Gráfico com os dados BRUTOS (Fundo levemente avermelhado)
            fig_antes = plotar_boxplot_comparativo(df_bruto, col, nome, "(Antes do Trimming)", "#fff5f5")
            st.pyplot(fig_antes)
            
        with col_direita:
            # Gráfico com os dados LIMPOS (Fundo levemente esverdeado/azulado)
            fig_depois = plotar_boxplot_comparativo(df_limpo, col, nome, "(Após o Trimming)", "#f5fbff")
            st.pyplot(fig_depois)

st.markdown("---")

# =====================================================================
# SEÇÃO 3: OTIMIZAÇÃO DE TIPOS E MEMÓRIA
# =====================================================================
st.header("3. Otimização Estrutural (Memory Downcast)")
st.write("A base de dados original consumia uma quantidade excessiva de memória RAM. Para viabilizar a execução fluida deste painel analítico na web, realizámos a alteração forçada dos tipos primitivos de dados e guardámos o ficheiro final no formato colunar `.parquet`.")

dados_tipos = {
    'Nome da Coluna': ['SG_UF_PROVA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'],
    'Tipo Original (.parquet bruto)': ['object', 'float64', 'float64', 'float64', 'float64', 'float64'],
    'Tipo Otimizado (.parquet limpo)': ['category', 'float32', 'float32', 'float32', 'float32', 'float32'],
    'Tamanho Novo (Por registo)': ['1 byte', '4 bytes', '4 bytes', '4 bytes', '4 bytes', '4 bytes']
}

df_tipos = pd.DataFrame(dados_tipos)
st.dataframe(df_tipos, use_container_width=True, hide_index=True)