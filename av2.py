import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

df = pd.read_csv('dataset_3.csv', sep=",")

def plotar_mapa_calor(df):
    plt.figure(figsize=(12, 10))
    correlacoes = df.corr(numeric_only=True)

    sb.heatmap(
        correlacoes,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10}
    )

    plt.title('Mapa de Calor - Correlação entre variáveis', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.show()

def plotar_tabela_estatisticas(df, colunas=None):
    
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
        'IQR': df.quantile(0.75) - df.quantile(0.25),
        'Qtd. Valores Nulos': df.isnull().sum()
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

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.2, 1.8)

    # Estilizando cabeçalhos
    for (row, col), cell in tabela.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f2f2f2")
        cell.set_edgecolor("#999999")

    plt.title("Estatísticas do DataFrame", fontsize=16, weight='bold', pad=10)
    plt.tight_layout()
    plt.show()

def plotar_tabela_classificatorias(df, colunas=None):
  
    if colunas is not None:
        df = df[colunas]
    else:
        df = df.select_dtypes(include=['object', 'category'])

    if df.empty:
        print("Nenhuma coluna categórica encontrada.")
        return

    estatisticas = {
        'Moda': df.mode().iloc[0],
        'Frequência da Moda': df.apply(lambda col: col.value_counts().iloc[0]),
        'Frequência Relativa da Moda (%)': df.apply(lambda col: round(100 * col.value_counts(normalize=True).iloc[0], 2)),
        'Qtd. Valores Únicos': df.nunique(),
        'Qtd. Valores Nulos': df.isnull().sum()
    }

    estat_df = pd.DataFrame(estatisticas).T

    largura = max(6, len(df.columns) * 2)
    altura = max(3, len(estat_df) * 0.7)

    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.axis('off')

    tabela = ax.table(
        cellText=estat_df.values,
        rowLabels=estat_df.index,
        colLabels=estat_df.columns,
        cellLoc='center',
        rowLoc='center',
        bbox=[0, 0, 1, 1]
    )

    tabela.auto_set_font_size(False)
    tabela.scale(1.2, 1.8)

    for (row, col), cell in tabela.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f2f2f2")
        cell.set_edgecolor("#999999")

    plt.title("Estatísticas de Variáveis Categóricas", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def pre_processamento_dados(df):
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
    df_dummies = pd.get_dummies(df, columns=colunas_categoricas, prefix_sep='_', dtype=int, drop_first=True)
    return df_dummies

def gerar_modelo_regrassao(df, colunas_mantidas=None):
    df_com_variaveis_dummy = transformar_classificatorias_em_dummies(df)
    
    if colunas_mantidas is not None:
        df_modelo = df_com_variaveis_dummy[colunas_mantidas + ['tempo_resposta']]
    else:
        df_modelo = df_com_variaveis_dummy

    plotar_mapa_calor(df_modelo)
    X = df_modelo.drop(columns=['tempo_resposta'])
    X = sm.add_constant(X)
    Y = df_modelo['tempo_resposta']
    modelo = sm.OLS(Y, X).fit()
    return modelo

def plotar_tabela_testes_t_modelo(modelo, titulo='Resumo do Modelo de Regressão', pvalor_limite=0.05):

    tabela = pd.DataFrame({
        'Coeficiente': modelo.params.round(4),
        'Erro Padrão': modelo.bse.round(4),
        't': modelo.tvalues.round(2),
        'P-valor': modelo.pvalues.round(4),
        'IC 95% Inf.': modelo.conf_int()[0].round(4),
        'IC 95% Sup.': modelo.conf_int()[1].round(4),
    })

    fig, ax = plt.subplots(figsize=(10, 0.2 * len(tabela) + 1))
    ax.axis('off')
    ax.axis('tight')

    table_plot = ax.table(
        cellText=tabela.values,
        colLabels=tabela.columns,
        rowLabels=tabela.index,
        cellLoc='center',
        loc='center'
    )

    table_plot.auto_set_font_size(True)
    table_plot.set_fontsize(10)
    table_plot.scale(1.1, 1.3)

    # Destaque nas células com p-valor > limite
    col_index_pvalor = list(tabela.columns).index('P-valor')
    for row_idx, pval in enumerate(tabela['P-valor']):
        if pval > pvalor_limite:
            cell = table_plot[(row_idx + 1, col_index_pvalor)] 
            cell.set_facecolor('#ffcccc') 

    plt.title(titulo, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plotar_tabela_test_f_modelo(modelo, titulo='Resumo Geral do Modelo'):

    resumo = {
        'R²': round(modelo.rsquared, 3),
        'R² Ajustado': round(modelo.rsquared_adj, 3),
        'Estatística F': round(modelo.fvalue, 2),
        'Prob (F-stat)': f"{modelo.f_pvalue:.2e}",
        'Log-Likelihood': round(modelo.llf, 2),
        'AIC': round(modelo.aic, 2),
        'BIC': round(modelo.bic, 2),
        'Nº de Observações': int(modelo.nobs),
        'Df Model': int(modelo.df_model),
        'Df Residual': int(modelo.df_resid),
        'Tipo de Covariância': modelo.cov_type,
    }

    tabela = pd.DataFrame(resumo.items(), columns=['Métrica', 'Valor'])

    fig, ax = plt.subplots(figsize=(6, 0.2 * len(tabela) + 1))
    ax.axis('off')
    ax.axis('tight')

    table_plot = ax.table(
        cellText=tabela.values,
        colLabels=tabela.columns,
        cellLoc='center',
        loc='center'
    )

    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)

    plt.title(titulo, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def teste_breusch_pagan(modelo):
    residuos = modelo.resid
    exog = modelo.model.exog
    bp_teste = het_breuschpagan(residuos, exog)
    bp_labels = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
    resultado_bp = dict(zip(bp_labels, bp_teste))
    return resultado_bp

# Função para plotar tabela do teste de Breusch-Pagan
def plotar_tabela_bp(resultado_bp):
    df_resultado = pd.DataFrame(resultado_bp.items(), columns=["Métrica", "Valor"])

    fig, ax = plt.subplots(figsize=(6, 2)) 
    ax.axis('off')  
    tabela = ax.table(cellText=df_resultado.values, colLabels=df_resultado.columns,
                      cellLoc='center', loc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.auto_set_column_width(col=list(range(len(df_resultado.columns))))

    plt.title("Resultado do Teste de Breusch-Pagan", fontsize=14, fontweight='bold')
    plt.show()

def plotar_residuos(modelo):
    residuos = modelo.resid
    valores_ajustados = modelo.fittedvalues

    sb.residplot(x=valores_ajustados, y=residuos, lowess=True, line_kws={'color': 'red'})
    plt.xlabel("Valores ajustados")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Ajustados")
    plt.axhline(0, color='grey', linestyle='--')
    plt.show()

# Descomente as linhas abaixo para gerar os gráficos e tabelas da analise exploratória dos dados sem o pre-processamento
# plotar_tabela_estatisticas(df)
# plotar_tabela_classificatorias(df)

pre_processamento_dados(df)

# Descomente as linhas abaixo para gerar os gráficos e tabelas da analise exploratória dos dados após o pre-processamento
# plotar_tabela_estatisticas(df)
# plotar_tabela_classificatorias(df)

modelo = gerar_modelo_regrassao(df)

plotar_tabela_test_f_modelo(modelo, titulo='Resumo do Modelo de Regressão Completo')
plotar_tabela_testes_t_modelo(modelo, titulo='Resumo do Modelo de Regressão Completo - Teste t')
plotar_residuos(modelo)
plotar_tabela_bp(teste_breusch_pagan(modelo))


modelo_reduzido = gerar_modelo_regrassao(df, ['ram_gb', 'cpu_cores'])
plotar_tabela_test_f_modelo(modelo_reduzido, titulo='Resumo do Modelo de Regressão Reduzido')
plotar_tabela_testes_t_modelo(modelo_reduzido, titulo='Resumo do Modelo de Regressão Reduzido - Teste t')
plotar_residuos(modelo_reduzido)
plotar_tabela_bp(teste_breusch_pagan(modelo_reduzido))







    
