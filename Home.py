import streamlit as st
import os

# Importando a função de limpeza do nosso script na pasta src
from src.pre_processing import executar_limpeza

st.set_page_config(page_title="Projeto ENEM 2023", page_icon="🎓", layout="centered")

# =====================================================================
# VERIFICAÇÃO AUTOMÁTICA E LIMPEZA DOS DADOS
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

# O "Pulo do Gato": Checa se a base limpa existe
if not os.path.exists(ARQUIVO_LIMPO):
    st.warning("⚠️ Arquivo otimizado não encontrado. Iniciando a limpeza da base original...")
    
    # Trava a tela com um carregamento enquanto roda o código pesado
    with st.spinner('Lendo "enem2023.parquet", removendo nulos e tratando outliers... Isso pode levar alguns minutos.'):
        try:
            executar_limpeza()
            st.success("✅ Limpeza concluída com sucesso! Recarregando a aplicação...")
            st.rerun() # Força a página a recarregar automaticamente após limpar
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop() # Para a aplicação se a base bruta não for encontrada

# =====================================================================
# PÁGINA INICIAL (HOME) DO SEU PROJETO
# =====================================================================
st.title("🎓 Observatório ENEM 2023")
st.markdown("### Uma Imersão Estatística: Estado do Paraná vs. Cenário Nacional")
st.markdown("---")

st.write("""
Este painel interativo foi desenvolvido como projeto central para a disciplina de **Processos Estocásticos**. 
O objetivo é realizar um diagnóstico profundo sobre o desempenho acadêmico e as correlações socioeconômicas 
dos candidatos do ENEM 2023, utilizando o estado do **Paraná** como ponto focal de comparação com o restante do **Brasil**.
""")

## 📌 Escopo do Projeto

st.markdown("""
A análise está estruturada em quatro pilares fundamentais, acessíveis pelo menu lateral:

1.  **👥 Perfil e Diversidade:** Exploração demográfica através de Pirâmides Etárias e distribuição de raça e gênero, identificando quem é o candidato paranaense.
2.  **📈 Desempenho Pedagógico:** Comparação de médias entre as áreas do conhecimento e uma análise detalhada das competências da Redação através de Gráficos de Radar.
3.  **💰 Impacto Socioeconômico:** Estudo estocástico sobre como a renda, o acesso à internet e a posse de tecnologia influenciam a curva de aprendizado e as notas finais.
4.  **🔬 Rigor Estatístico:** Demonstração técnica do tratamento de dados, análise de normalidade (QQ-Plots) e modelagem de resíduos para correção de heterocedasticidade.
""")

st.markdown("---")

st.markdown("### 🛠️ Metodologia Aplicada")
st.markdown("> **Limpeza Estocástica:** Aplicação de *Trimming* via Intervalo Interquartil (IQR) para remoção de outliers, garantindo amostras com variância estável para os modelos preditivos.")

st.info("""
**Autores:** 
* Victor Hugo Aló 
* Guilherme Albuquerque Zaparolli 
* Pedro Henrique Silveira Stuckus Pintor  

**Orientação:** Professor César Candido Xavier  
**Instituição:** Universidade de Sorocaba (UNISO)
""")

st.write("---")
st.caption("👈 Utilize o menu lateral para iniciar a exploração dos dados.")