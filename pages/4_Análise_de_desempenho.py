import streamlit as st
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Análise de Desempenho", page_icon="📈", layout="wide")

st.title("📈 Análise de Desempenho: Áreas, Línguas e Redação")
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
# CARREGAMENTO DOS DADOS (ULTRA-OTIMIZADO)
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
    
    # --- MEMORY DOWNCAST (Previne o travamento da RAM no Servidor) ---
    df['SG_UF_PROVA'] = df['SG_UF_PROVA'].astype('category')
    df['Regiao'] = df['Regiao'].astype('category')
    df['TP_LINGUA'] = df['TP_LINGUA'].astype('category')
    
    colunas_float = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',
                     'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
    for col in colunas_float:
        df[col] = df[col].astype('float32')
        
    return df

with st.spinner("Carregando base de desempenho..."):
    df = carregar_dados_desempenho()

colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
nomes_areas = ['C. Natureza', 'C. Humanas', 'Linguagens', 'Matemática', 'Redação']

# =====================================================================
# FUNÇÕES DE PLOTAGEM (PLOTLY COM PRÉ-CÁLCULO)
# =====================================================================
def gerar_barras_medias():
    df_medias = df.groupby('Regiao')[colunas_notas].mean().T
    df_medias.index = nomes_areas
    
    fig = go.Figure()
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        fig.add_trace(go.Bar(
            name=regiao, x=df_medias.index, y=df_medias[regiao],
            marker_color=cor, text=[f"{v:.1f}" for v in df_medias[regiao]], textposition='outside'
        ))
        
    fig.update_layout(title="Média de Notas por Área do Conhecimento", barmode='group', yaxis_title="Nota Média", margin=dict(t=50, b=20, l=20, r=20))
    return fig

def gerar_barras_std():
    df_std = df.groupby('Regiao')[colunas_notas].std().T
    df_std.index = nomes_areas
    
    fig = go.Figure()
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        fig.add_trace(go.Bar(
            name=regiao, x=df_std.index, y=df_std[regiao],
            marker_color=cor, text=[f"{v:.1f}" for v in df_std[regiao]], textposition='outside'
        ))
        
    fig.update_layout(title="Dispersão (Desvio Padrão) por Área", barmode='group', yaxis_title="Desvio Padrão", margin=dict(t=50, b=20, l=20, r=20))
    return df_std, fig

def gerar_barras_lingua():
    mapa_lingua = {0: 'Inglês', 1: 'Espanhol'}
    df_lingua = df.copy()
    df_lingua['Língua'] = df_lingua['TP_LINGUA'].map(mapa_lingua)
    
    ct_lingua = pd.crosstab(df_lingua['Regiao'], df_lingua['Língua'], normalize='index') * 100
    
    fig = go.Figure()
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        if regiao in ct_lingua.index:
            y_vals = [ct_lingua.loc[regiao, 'Inglês'], ct_lingua.loc[regiao, 'Espanhol']]
            fig.add_trace(go.Bar(
                name=regiao, x=['Inglês', 'Espanhol'], y=y_vals,
                marker_color=cor, text=[f"{v:.1f}%" for v in y_vals], textposition='outside'
            ))
            
    fig.update_layout(title="Proporção de Escolha: Inglês vs Espanhol", barmode='group', yaxis_title="Percentual (%)")
    return fig

def gerar_boxplot_lingua_estatistico():
    fig = go.Figure()
    mapa_lingua = {0: 'Inglês', 1: 'Espanhol'}
    
    for regiao, cor in [('Paraná (PR)', '#1f77b4'), ('Brasil (Sem PR)', '#ff7f0e')]:
        for lingua_codigo, lingua_nome in [(0, 'Inglês'), (1, 'Espanhol')]:
            # Filtro e amostragem para o Boxplot não travar (Reduzido para 10.000 para fluidez máxima)
            mask = (df['Regiao'] == regiao) & (df['TP_LINGUA'] == lingua_codigo)
            data = df[mask]['NU_NOTA_LC'].dropna()
            
            if len(data) > 10000: data = data.sample(10000, random_state=42)
            
            if not data.empty:
                fig.add_trace(go.Box(
                    y=data, name=lingua_nome, legendgroup=regiao,
                    marker_color=cor, showlegend=(lingua_codigo == 0),
                    line=dict(color=cor), fillcolor=cor, opacity=0.7,
                    hovertemplate=f"<b>{regiao} - {lingua_nome}</b><br>Nota: %{{y}}<extra></extra>"
                ))
                
    fig.update_layout(title="Notas de Linguagens (LC) por Língua", boxmode='group', yaxis_title="Nota LC", legend_title="Região")
    return fig

def gerar_radar_competencias():
    col_comp = ['NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
    labels_comp = ['Gramática', 'Tema/Repertório', 'Argumentação', 'Coesão', 'Proposta Interv.']
    
    df_comp = df.groupby('Regiao')[col_comp].mean().T
    df_comp.index = labels_comp
    
    # "Pulo do Gato" para zoom dinâmico
    min_nota = df_comp.min().min()
    limite_inferior = max(0, min_nota - 15)
    
    fig = go.Figure()

    # Configuração de Cores (Linha Sólida, Preenchimento Transparente)
    config_cores = [
        ('Paraná (PR)', '#1f77b4', 'rgba(31, 119, 180, 0.4)'), 
        ('Brasil (Sem PR)', '#ff7f0e', 'rgba(255, 127, 14, 0.3)') # Brasil um pouco mais transparente para destacar o PR
    ]

    for regiao, cor_linha, cor_preenchimento in config_cores:
        if regiao in df_comp.columns:
            valores = df_comp[regiao].tolist()
            # O Plotly exige que o gráfico feche o círculo repetindo o primeiro valor
            valores += valores[:1]
            theta_labels = labels_comp + [labels_comp[0]]

            fig.add_trace(go.Scatterpolar(
                r=valores, 
                theta=theta_labels, 
                fill='toself', 
                name=regiao,
                line=dict(color=cor_linha, width=3),
                fillcolor=cor_preenchimento,
                marker=dict(size=8, color=cor_linha)
            ))
        
    fig.update_layout(
        title=dict(
            text="Radar de Competências da Redação<br><sup>(Escala ajustada para realçar diferenças estocásticas)</sup>", 
            x=0.5, 
            font=dict(size=18, color='#e0e0e0') # Título em cinza claro para contraste
        ),
        # TRANSPARÊNCIA DO FUNDO
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, 
                range=[limite_inferior, 160], 
                gridcolor="rgba(255, 255, 255, 0.15)", # Linhas circulares sutis
                tickfont=dict(size=10, color='#b0b0b0'),
                angle=30 # Rotaciona os números das notas para não ficarem em cima dos eixos
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#ffffff'), # Labels externas em branco
                gridcolor="rgba(255, 255, 255, 0.15)"    # Linhas dos eixos (raios)
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='rgba(0,0,0,0)',
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.5
        ),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    return df_comp, fig

# =====================================================================
# RENDERIZAÇÃO DA PÁGINA
# =====================================================================

# --- 1. ÁREAS DE MAIOR DESEMPENHO ---
st.header("1. Áreas de maior desempenho?")
st.write("Comparação das médias aritméticas das notas nas 5 áreas do conhecimento.")
fig_medias = obter_grafico_cache("bar_medias_areas.json", gerar_barras_medias)
st.plotly_chart(fig_medias, use_container_width=True)

st.markdown("---")

# --- 2. DISPERSÃO ---
st.header("2. Área de maior dispersão?")
st.write("O Desvio Padrão indica o nível de desigualdade: quanto maior o valor, mais heterogêneo é o desempenho dos alunos.")
col_esc, col_dir = st.columns([1, 2])

with st.spinner("Processando..."):
    df_std, fig_std = gerar_barras_std() # Calculamos dinâmico para popular a tabela

with col_esc:
    st.dataframe(df_std.style.format("{:.2f}"), use_container_width=True)
with col_dir:
    fig_std_cache = obter_grafico_cache("bar_std_areas.json", gerar_barras_std)[1] if not 'fig_std' in locals() else fig_std
    st.plotly_chart(fig_std_cache, use_container_width=True)

st.markdown("---")

# --- 3. LÍNGUA ESTRANGEIRA ---
st.header("3. Escolha de Língua Estrangeira")
col_a, col_b = st.columns(2)

with col_a:
    st.write("**Proporção de Escolha**")
    fig_lingua_prop = obter_grafico_cache("bar_lingua_proporcao.json", gerar_barras_lingua)
    st.plotly_chart(fig_lingua_prop, use_container_width=True)

with col_b:
    st.write("**Desempenho (Boxplot Amostral)**")
    fig_lingua_box = obter_grafico_cache("box_lingua_notas.json", gerar_boxplot_lingua_estatistico)
    st.plotly_chart(fig_lingua_box, use_container_width=True)

st.markdown("---")

# --- 4. REDAÇÃO (RADAR) ---
st.header("4. Competências da Redação")
st.write("Análise das 5 competências avaliadas na Redação (0 a 200 pontos cada).")

with st.spinner("Processando Radar..."):
    df_radar, fig_radar = gerar_radar_competencias()

st.dataframe(df_radar.style.format("{:.2f}"), use_container_width=True)

fig_radar_cache = obter_grafico_cache("radar_redacao.json", lambda: gerar_radar_competencias()[1])
st.plotly_chart(fig_radar_cache, use_container_width=True)