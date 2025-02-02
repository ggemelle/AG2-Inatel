import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def carregar_df():
    df = pd.read_csv('wholesale.csv')
    # seleciona as colunas de atributos e a coluna alvo
    x = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
    y = df['Channel'] - 1  # ajusta para rótulos 0 e 1
    return x, y

def escolher_modelo():
    """Permite ao usuário escolher entre KNN ou Decision Tree."""
    while True:
        print("\nEscolha o modelo de classificação:")
        print("1. k-Nearest Neighbors (kNN)")
        print("2. Decision Tree")
        escolha = input("Digite o número da sua escolha: ").strip()

        if escolha == "1":
            return KNeighborsClassifier(), "k-Nearest Neighbors"
        elif escolha == "2":
            return DecisionTreeClassifier(), "Decision Tree"
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

def main():
    print("\nBem-vindo ao Classificador de Canais de Vendas!")
    print("Aqui identificaremos o tipo de cliente: HoReCa ou Retail.\n")

    x, y = carregar_df()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Escolha do modelo
    modelo, nome_modelo = escolher_modelo()
    print(f"\nModelo escolhido: {nome_modelo}")

    # Treinando o modelo
    modelo.fit(x_train, y_train)
    y_predict = modelo.predict(x_test)

    print("\nMétricas de Avaliação:")
    print(classification_report(y_test, y_predict))

    # nomes das colunas de atributos
    nomes_colunas = x.columns

    while True:
        entrada = input("\nDigite os valores para [Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen] separados por vírgula (ou 'sair' para encerrar): ").strip()
        if entrada.lower() == 'sair':
            print("Saindo!")
            break
        else:
            try:
                # convertendo entrada em um DataFrame com os nomes das colunas
                valores = np.array([float(x) for x in entrada.split(',')]).reshape(1, -1)
                dados_usuario = pd.DataFrame(valores, columns=nomes_colunas)

                # realizar a previsão
                resultado = modelo.predict(dados_usuario)[0]
                tipo_cliente = "HoReCa" if resultado == 0 else "Retail"
                print(f"Classificação: {tipo_cliente}")
            except ValueError:
                print("Erro: Insira os valores no formato correto (números separados por vírgula).")
            except Exception as e:
                print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
