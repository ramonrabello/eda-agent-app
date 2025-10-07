import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_quick_eda(df, analysis_type):
    figs = []
    if analysis_type == "Descrição":
        description = df.describe().T
        return ("Estatísticas descritivas:", [] if description.empty else [sns.heatmap(description.corr())])
    elif analysis_type == "Correlação":
        plt.figure(figsize=(10,8))
        fig = sns.heatmap(df.corr(), annot=False)
        figs.append(fig.figure)
        return ("Heatmap de correlação:", figs)
    elif analysis_type == "Outliers":
        plt.figure()
        fig = df.boxplot()
        figs.append(fig.figure)
        return ("Boxplot geral para detecção de outliers.", figs)
    elif analysis_type == "Distribuição":
        plt.figure()
        fig = df['Amount'].hist(bins=50)
        figs.append(fig.figure)
        return ("Distribuição dos valores de transação.", figs)
    elif analysis_type == "Classes":
        plt.figure()
        fig = df['Class'].value_counts().plot.pie(autopct='%.2f%%')
        figs.append(fig.figure)
        return ("Distribuição das classes.", figs)
    else:
        return ("Análise não reconhecida.", [])

def plot_eda_figures(df):
    # Exemplo de função utilitária para desenhar figuras adicionais
    pass
