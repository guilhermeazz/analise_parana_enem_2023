import streamlit as st
import pandas as pd
import os

# Importando a função de limpeza do nosso script na pasta src
from src.pre_processing import executar_limpeza

st.set_page_config(page_title="Observatório ENEM 2023", page_icon="🎓", layout="wide")

# =====================================================================
# VERIFICAÇÃO AUTOMÁTICA E LIMPEZA DOS DADOS
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

# O "Pulo do Gato": Checa se a base limpa existe
if not os.path.exists(ARQUIVO_LIMPO):
    st.warning("⚠️ Arquivo otimizado não encontrado. Iniciando a limpeza da base original...")
    
    with st.spinner('Lendo "enem2023.parquet", removendo nulos e tratando outliers... Isso pode levar alguns minutos.'):
        try:
            executar_limpeza()
            st.success("✅ Limpeza concluída com sucesso! Recarregando a aplicação...")
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

# =====================================================================
# FUNÇÃO DE MÉTRICAS RÁPIDAS (LEITURA ULTRA-LEVE)
# =====================================================================
@st.cache_data
def carregar_metricas_iniciais():
    """Lê apenas a coluna de estado para não pesar a memória na Home"""
    if os.path.exists(ARQUIVO_LIMPO):
        df_uf = pd.read_parquet(ARQUIVO_LIMPO, columns=['SG_UF_PROVA'])
        total = len(df_uf)
        pr_total = (df_uf['SG_UF_PROVA'] == 'PR').sum()
        return total, pr_total
    return 0, 0

total_candidatos, total_pr = carregar_metricas_iniciais()

# =====================================================================
# PÁGINA INICIAL (HOME) DO SEU PROJETO
# =====================================================================

# Cabeçalho Principal com design limpo
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🎓 Observatório ENEM 2023</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Uma Imersão Estatística: Estado do Paraná vs. Cenário Nacional</h4>", unsafe_allow_html=True)
st.markdown("---")

# Introdução
st.write("""
Este painel interativo foi desenvolvido como projeto central para a disciplina de **Processos Estocásticos**. 
O objetivo é realizar um diagnóstico profundo sobre o desempenho acadêmico e as correlações socioeconômicas 
dos candidatos do ENEM 2023, utilizando ferramentas de *Data Science* para comparar o estado do **Paraná** com a média do **Brasil**.
""")

# Métricas de Impacto Visual
if total_candidatos > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume de Dados Analisados", f"{total_candidatos:,}".replace(',', '.'))
    c2.metric("Amostra do Paraná (PR)", f"{total_pr:,}".replace(',', '.'))
    c3.metric("Variáveis em Estudo", "Exploração Cruzada")
    st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# ESCOPO DO PROJETO EM GRELHA (CARDS)
# =====================================================================
st.markdown("### 📌 Escopo da Análise")
st.write("Aceda ao menu lateral para navegar entre os quatro pilares fundamentais deste estudo:")

# Criando um grid 2x2 para os pilares
col1, col2 = st.columns(2)

with col1:
    st.info("**👥 Perfil e Diversidade**\n\nExploração demográfica através de Pirâmides Etárias e distribuição de raça e gênero, identificando quem é o candidato paranaense.")
    st.success("**💰 Impacto Socioeconômico**\n\nEstudo estocástico sobre como a renda, o acesso à internet e a posse de tecnologia influenciam a curva de aprendizado e as notas finais.")

with col2:
    st.warning("**📈 Desempenho Pedagógico**\n\nComparação de médias entre as áreas do conhecimento e uma análise detalhada das competências da Redação através de Gráficos de Radar.")
    st.error("**🔬 Rigor Estatístico**\n\nDemonstração técnica do tratamento de dados, análise de normalidade (QQ-Plots) e cruzamento vetorial para detecção de anomalias.")

st.markdown("---")

# =====================================================================
# METODOLOGIA E CRÉDITOS
# =====================================================================
# Escondendo a parte muito técnica num expansor para deixar a Home mais limpa
with st.expander("🛠️ Ver Metodologia Aplicada"):
    st.write("""
    **Engenharia de Dados e Limpeza Estocástica:**
    * Aplicação de *Trimming* via Intervalo Interquartil (IQR) para remoção de *outliers*.
    * Otimização de tipos primitivos (Memory Downcast) no formato `.parquet` para renderização web de alta performance.
    * Processamento vetorial utilizando Pandas e visualizações interativas via Plotly.
    """)

# Créditos bem organizados
st.markdown("<br>", unsafe_allow_html=True)
c_autores, c_inst = st.columns(2)

with c_autores:
    st.markdown("**Autores do Projeto:**")
    st.markdown("👨‍💻 Guilherme Albuquerque Zaparolli")
    st.markdown("👨‍💻 Victor Hugo Aló")
    st.markdown("👨‍💻 Pedro Henrique Silveira Stuckus Pintor")

with c_inst:
    st.markdown("**Orientação Acadêmica:**")
    st.markdown("🎓 Professor César Candido Xavier")
    st.markdown("🏛️ Universidade de Sorocaba (UNISO)")

st.markdown("---")
st.caption("👈 Utilize o menu lateral para iniciar a exploração do Observatório.")