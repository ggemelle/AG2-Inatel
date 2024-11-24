import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def carregar_df():
    df = pd.read_csv('wholesale.csv')
    x = df.drop(columns=['Channel'])
    y = df['Channel'] - 1  # ajusta para 0 e 1
    return x, y

def main():
    print("\nBem-vindo! Vamos classificar canais de venda (HoReCa ou Retail).")

    # separando o conjunto de dados
    x, y = carregar_df()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # usando o modelo kNN
    modelo = KNeighborsClassifier()
    modelo.fit(x_train, y_train)
    y_predict = modelo.predict(x_test)

    print("\nMétricas de Avaliação:")
    print(classification_report(y_test, y_predict))

    while True:
        entrada = input("\nDigite os valores separados por vírgula (ou 'sair' para encerrar): ").strip()
        if entrada.lower() == 'sair':
            print("Encerrando o programa. Até mais!")
            break
        else:
            try:
                valores = np.array([float(x) for x in entrada.split(',')]).reshape(1, -1)
                resultado = modelo.predict(valores)[0]
                print(f"Resultado: {'HoReCa' if resultado == 0 else 'Retail'}")
            except ValueError:
                print("Erro nos valores. Insira os dados corretamente.")

if __name__ == "__main__":
    main()
