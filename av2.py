import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm 

df = pd.read_csv('dataset_3.csv', sep=",")

def plotarMapaDeCalor(df):
    sb.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='inferno')
    plt.title('Mapa de calor')
    plt.show()

def plotar_tabela_estatisticas(df, colunas=None):
    """
    Plota uma tabela estilizada com estatísticas básicas do DataFrame.
    """

    if colunas is not None:
        df = df[colunas]
    else:
        df = df.select_dtypes(include='number')

    if df.empty:
        print("Nenhuma coluna numérica encontrada.")
        return

    estatisticas = {
        'Média': df.mean(),
        'Mediana': df.median(),
        'Desvio Padrão': df.std(),
        'Mínimo': df.min(),
        'Máximo': df.max(),
        'Q1': df.quantile(0.25),
        'Q3': df.quantile(0.75),
        'IQR': df.quantile(0.75) - df.quantile(0.25)
    }

    estat_df = pd.DataFrame(estatisticas).T.round(2)

    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.5 + 1, 5))
    ax.axis('off')

    tabela = ax.table(
        cellText=estat_df.values,
        rowLabels=estat_df.index,
        colLabels=estat_df.columns,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    tabela.auto_set_font_size(True)
    tabela.scale(1.2, 1.8)

    # Estilizando cabeçalhos
    for (row, col), cell in tabela.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f2f2f2")
        cell.set_edgecolor("#999999")

    plt.title("Estatísticas do DataFrame", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def preProcessamentoDosDados(df):
    colunas = df.columns
    for coluna in colunas:
        if df[coluna].isnull().sum() > 0:
            # Verifica se a coluna é categórica (object) ou numérica
            # Se for categórica, preenche com a moda (valor mais frequente)
            if df[coluna].dtype == 'object':
                df[coluna].fillna(df[coluna].mode()[0], inplace=True)
            else:
                mediana = df[coluna].median()
                # Se for numérica, preenche com a mediana
                df[coluna].fillna(mediana, inplace=True)
                
    return df

def transformar_classificatorias_em_dummies(df):
   
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns
    df_dummies = pd.get_dummies(df, columns=colunas_categoricas, prefix_sep='_', dtype=int)
    return df_dummies



def gerarModeloRegrassao(df):
    df_com_variaveis_dummy = transformar_classificatorias_em_dummies(df)
    print(df_com_variaveis_dummy.columns)
    plotarMapaDeCalor(df_com_variaveis_dummy)
    X = df_com_variaveis_dummy.drop(columns=['tempo_resposta'])
    X = sm.add_constant(X)
    Y = df_com_variaveis_dummy['tempo_resposta']
    modelo = sm.OLS(Y, X).fit()
    return modelo



# print("Valores nulos por coluna após o preenchimento:")
df_atualizado = preProcessamentoDosDados(df)

modelo = gerarModeloRegrassao(df_atualizado)

print(modelo.summary())
# print(df_atualizado.isnull().sum()) 

#plotar_tabela_estatisticas(df_atualizado)






    
