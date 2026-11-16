import numpy as np
import pandas as pd

filePath = "imoveis_lar_perfeito.csv"

def calcular_regressao_linear(filePath):

    df = pd.read_csv(filePath)

    area = df['area_util_m2'].to_list()
    quartos = df['quantidade_quartos'].to_list()
    preco = df['preco_venda'].to_list()

    n = len(preco)

    X = np.column_stack((np.ones(n), area, quartos))
    y = np.array(preco)
    
    XT = np.transpose(X)
    XTy = np.dot(XT, y)
    XTXinv = np.linalg.inv(np.dot(XT, X))

    beta = np.dot(XTXinv, XTy)
    beta = np.round(beta, 2)

    return beta.tolist()

resultado = calcular_regressao_linear(filePath)
print("Equação do Modelo:")
print(f"Preco = {resultado[0]} + {resultado[1]} * Area + {resultado[2]} * Quartos")