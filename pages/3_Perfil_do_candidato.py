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
ARQUIVO_DADOS = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

st.set_page_config(page_title="Perfil do Candidato", page_icon="👥", layout="wide")

st.title("👥 Perfil Sociodemográfico Comparativo: Paraná vs Brasil")
st.markdown("---")

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
    
    # Otimiza tipos
    df_perfil['TP_SEXO'] = df_perfil['TP_SEXO'].astype('category')
    df_perfil['TP_ESCOLA'] = pd.to_numeric(df_perfil['TP_ESCOLA'], errors='coerce')
    
    # Cria a coluna de Região
    df_perfil['Regiao'] = np.where(df_perfil['SG_UF_PROVA'] == 'PR', 'Paraná (PR)', 'Brasil (Sem PR)')
    
    return df_perfil

with st.spinner("A carregar e processar o perfil sociodemográfico..."):
    df = carregar_dados_perfil()

# =====================================================================
# FUNÇÕES DE PLOTAGEM DE BI (BARRAS LADO A LADO)
# =====================================================================

def plotar_barras_agrupadas_percentual(df_subset, coluna, dicionario_mapa, titulo, subtitulo, rotacao_x=0):
    """Gera um gráfico de barras verticais agrupadas (PR Azul e BR Laranja lado a lado)"""
    df_plot = df_subset.copy()
    df_plot[coluna] = df_plot[coluna].map(dicionario_mapa)
    
    # Cria a tabela percentual normalizando por coluna (Região) para somar 100% em cada uma
    ct = pd.crosstab(df_plot[coluna], df_plot['Regiao'], normalize='columns') * 100
    
    # Garante que o PR (Azul) venha primeiro e o Brasil (Laranja) depois
    ct = ct[['Paraná (PR)', 'Brasil (Sem PR)']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plota as barras não empilhadas (grouped), definindo as cores estritas
    ct.plot(kind='bar', stacked=False, ax=ax, color=['#1f77b4', '#ff7f0e'], alpha=0.9, width=0.7)
    
    ax.set_title(f'{titulo}\n{subtitulo}', weight='bold')
    ax.set_ylabel('Percentual dentro da Região (%)')
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=rotacao_x)
    
    # Rótulos no topo de cada barra
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # Para evitar poluição de rótulos com 0%
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', color='black', weight='bold', fontsize=9)
            
    ax.legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plotar_piramide_etaria_percentual(df_subset, titulo_regiao, cor_masc, cor_fem):
    """Gera uma pirâmide etária com idades mais jovens embaixo e mais velhas no topo"""
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Lista de labels na ordem original (mais novo para mais velho)
    labels_idade = [
        '< 17', '17', '18', '19', '20', '21', '22', '23', '24', '25',
        '26 a 30', '31 a 35', '36 a 40', '41 a 45', '46 a 50', 
        '51 a 55', '56 a 60', '61 a 65', '66 a 70', '> 70'
    ]
    
    # Para o gráfico colocar o mais novo embaixo, precisamos INVERTER a lista de ordem
    labels_invertidos = labels_idade[::-1]
    
    # Dicionário de mapeamento: 1 vira '< 17', 2 vira '17', etc.
    mapa_idade = dict(enumerate(labels_idade, 1))
    
    df_temp = df_subset.dropna(subset=['TP_FAIXA_ETARIA', 'TP_SEXO']).copy()
    df_temp['Faixa Etária'] = df_temp['TP_FAIXA_ETARIA'].map(mapa_idade)
    
    tamanho_total = len(df_temp)
    
    df_male = df_temp[df_temp['TP_SEXO'] == 'M']
    df_female = df_temp[df_temp['TP_SEXO'] == 'F']
    
    # Calcula as proporções
    male_dist = (df_male['Faixa Etária'].value_counts() / tamanho_total) * 100
    female_dist = (df_female['Faixa Etária'].value_counts() / tamanho_total) * 100
    
    # Reindexa garantindo que todas as idades apareçam, inverte o masculino para o lado esquerdo (-)
    male_dist = male_dist.reindex(labels_invertidos).fillna(0) * -1
    female_dist = female_dist.reindex(labels_invertidos).fillna(0)
    
    # Plota as barras horizontais
    sns.barplot(x=male_dist.values, y=male_dist.index, order=labels_invertidos, color=cor_masc, ax=ax, label='Masculino')
    sns.barplot(x=female_dist.values, y=female_dist.index, order=labels_invertidos, color=cor_fem, ax=ax, label='Feminino')
    
    ax.set_title(f'Pirâmide Etária: {titulo_regiao}', weight='bold')
    ax.set_xlabel('Percentual do Total da Região (%)')
    ax.set_ylabel('Faixa Etária (Anos)')
    
    # Formata o eixo X para que não mostre números negativos
    ticks = plt.xticks()[0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{abs(t):.0f}%' for t in ticks])
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    return fig

# =====================================================================
# RENDERIZAÇÃO DAS PERGUNTAS DE ANÁLISE
# =====================================================================

# --- PERGUNTA 1: PERFIL DO CANDIDATO ---
st.header("1. Qual o perfil sociodemográfico do candidato?")
st.write("Analisamos a distribuição de sexo, raça/cor e idade para entender as características populacionais dos inscritos, comparando o Paraná com a média Nacional.")

tab1, tab2 = st.tabs(["Distribuição por Sexo", "Distribuição por Raça/Cor"])

with tab1:
    st.write("**Distribuição Percentual por Sexo (Masculino vs Feminino)**")
    mapa_sexo = {'M': 'Masculino', 'F': 'Feminino'}
    # Passando rotata_x=0 para que os nomes fiquem retos
    fig_sexo = plotar_barras_agrupadas_percentual(df, 'TP_SEXO', mapa_sexo, 'Perfil Comparativo: Sexo', '(Comparação Proporcional)', rotacao_x=0)
    st.pyplot(fig_sexo)

with tab2:
    st.write("**Distribuição Percentual por Raça/Cor Declarada**")
    mapa_raca = {
        0: 'Não Declarado', 1: 'Branca', 2: 'Preta', 
        3: 'Parda', 4: 'Amarela', 5: 'Indígena'
    }
    # Passando rotata_x=45 para os rótulos não se sobreporem
    fig_raca = plotar_barras_agrupadas_percentual(df, 'TP_COR_RACA', mapa_raca, 'Perfil Comparativo: Raça/Cor', '(Comparação Proporcional)', rotacao_x=45)
    st.pyplot(fig_raca)

st.markdown("<br>", unsafe_allow_html=True)

# PIRÂMIDES ETÁRIAS (COM CORES NO PADRÃO ESTABELECIDO)
st.write("**Pirâmide Etária Comparativa (Distribuição Etária por Sexo)**")
col_esq, col_dir = st.columns(2)

with col_esq:
    st.write("*Visualização do Paraná (Tons de Azul)*")
    # Tons de azul: Masculino mais escuro, Feminino mais claro
    fig_pir_pr = plotar_piramide_etaria_percentual(df[df['Regiao'] == 'Paraná (PR)'], 'Paraná (PR)', cor_masc='#08519c', cor_fem='#3182bd')
    st.pyplot(fig_pir_pr)

with col_dir:
    st.write("*Visualização do Brasil (Tons de Laranja)*")
    # Tons de laranja: Masculino mais escuro, Feminino mais claro
    fig_pir_br = plotar_piramide_etaria_percentual(df[df['Regiao'] == 'Brasil (Sem PR)'], 'Brasil (Sem PR)', cor_masc='#d94801', cor_fem='#fd8d3c')
    st.pyplot(fig_pir_br)

st.markdown("---")

# --- PERGUNTA 2: TREINEIROS ---
st.header("2. Qual a taxa de treineiros?")
st.write("A taxa de treineiros mede a proporção de inscritos que ainda não concluíram o Ensino Médio e realizam a prova apenas para autoavaliação.")

mapa_treineiro = {1: 'Treineiro', 0: 'Candidato Regular'}
fig_treineiro = plotar_barras_agrupadas_percentual(df, 'IN_TREINEIRO', mapa_treineiro, 'Taxa Comparativa de Treineiros', '(Comparação Proporcional)', rotacao_x=0)
st.pyplot(fig_treineiro)

st.markdown("---")

# --- PERGUNTA 3: ESCOLAS PÚBLICAS VS PRIVADAS ---
st.header("3. Qual a proporção de alunos de escolas públicas e privadas?")
st.write("Analisamos a origem escolar dos candidatos que declararam sua situação no momento da inscrição.")

mapa_escola = {2: 'Escola Pública', 3: 'Escola Privada'}
# Filtra apenas os candidatos válidos para criar um subset
df_escola = df[df['TP_ESCOLA'].isin([2, 3])].copy()

# Reutilizamos a função agrupada passando o subset
fig_escola = plotar_barras_agrupadas_percentual(df_escola, 'TP_ESCOLA', mapa_escola, 'Proporção de Alunos: Pública vs Privada', '(Percentual entre Declarantes)', rotacao_x=0)
st.pyplot(fig_escola)