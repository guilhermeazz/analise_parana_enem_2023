import pandas as pd
import os

# =====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E ARQUIVOS
# =====================================================================
# Mapeia o caminho: volta uma pasta atrás (de src/ para a raiz) e entra em data/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Garante que a pasta data/ existe
os.makedirs(DATA_DIR, exist_ok=True)

# Define os caminhos dos ficheiros
ARQUIVO_BRUTO = os.path.join(DATA_DIR, 'enem2023.parquet') 
ARQUIVO_LIMPO = os.path.join(DATA_DIR, 'enem_2023_limpo.parquet')

def executar_limpeza():
    print("A iniciar a limpeza automatizada dos dados...")
    
    # Verifica se o ficheiro bruto existe antes de tentar limpar
    if not os.path.exists(ARQUIVO_BRUTO):
        raise FileNotFoundError(f"ERRO: O ficheiro original '{ARQUIVO_BRUTO}' não foi encontrado na pasta 'data'. Por favor, coloque lá o ficheiro e tente novamente.")

    # 1. Carregar a base bruta
    print("A ler o ficheiro original...")
    df_bruto = pd.read_parquet(ARQUIVO_BRUTO)

    # 2. Remover nulos (Dropna)
    print("A remover os dados faltantes (abstenções e notas em branco)...")
    colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    df_limpo = df_bruto.dropna(subset=colunas_notas).copy()

    # 3. Tratamento de Outliers (Trimming / Truncamento via IQR)
    print("A aplicar a técnica de Trimming para remoção de outliers...")
    for col in colunas_notas:
        Q1 = df_limpo[col].quantile(0.25)
        Q3 = df_limpo[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        # Filtra mantendo apenas os dados dentro dos limites
        df_limpo = df_limpo[(df_limpo[col] >= limite_inf) & (df_limpo[col] <= limite_sup)]

    # 4. Otimização de Tipos de Dados (Memory Downcast)
    print("A otimizar a estrutura de dados para poupar memória RAM...")
    for col in colunas_notas:
        df_limpo[col] = pd.to_numeric(df_limpo[col], downcast='float')
    df_limpo['SG_UF_PROVA'] = df_limpo['SG_UF_PROVA'].astype('category')

    # 5. Guardar os resultados
    print(f"A guardar a base final limpa e otimizada em: {ARQUIVO_LIMPO}")
    df_limpo.to_parquet(ARQUIVO_LIMPO, index=False)

    print("\n✅ Processo de ETL concluído com sucesso! A aplicação já pode ser utilizada.")

# Permite que o script seja executado isoladamente no terminal
if __name__ == "__main__":
    executar_limpeza()