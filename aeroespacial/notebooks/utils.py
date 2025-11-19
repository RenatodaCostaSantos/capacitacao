from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def criar_modelo_regularizado(
    neuronios_entrada, neurons=45, learning_rate=0.001, dropout_rate=0.3
):
    """Cria e compila um modelo de Rede Neural com Dropout."""
    tf.random.set_seed(42)
    np.random.seed(42)

    tf.keras.backend.clear_session()  # força o TensorFlow a liberar recursos de sessões antigas.

    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential(
        [
            # 1. Camada de Entrada
            Input(shape=(neuronios_entrada,)),
            Dense(neurons, activation="relu"),
            # 2. DROPOUT (Descarta 30% dos neurônios aleatoriamente)
            Dropout(dropout_rate),
            # 3. Segunda Camada Oculta
            Dense(32, activation="relu"),
            # 4. DROPOUT (Ajuda a evitar que a segunda camada decore)
            Dropout(dropout_rate),
            # 5. Camada de Saída
            Dense(1, activation="linear"),
        ]
    )

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def cria_feature_media_ponderada(df):
    """
    Cria uma feature fazendo-se a média ponderada dos coeficientes do espectro de potencias da CMB.
    """
    l_start = 2
    l_end = 511

    # Vetor l (o valor do momento de multipolo)
    l_values = np.arange(l_start, l_end + 1)

    # Vetor de pesos: l * (l + 1)
    weights = l_values * (l_values + 1) / 2 * np.pi

    # 4. Denominador da média ponderada (soma dos pesos)
    sum_of_weights = weights.sum()

    # 5. Cálculo do Numerador e da Nova Feature
    # O Pandas alinha as colunas de Cl_data com o vetor 'weights' e multiplica.
    # Depois, .sum(axis=1) soma o resultado dessa multiplicação ao longo das colunas,
    # resultando no numerador para CADA linha (observação).
    numerator = (df * weights).sum(axis=1)

    # Média ponderada
    df["media_ponderada_Cls"] = numerator / sum_of_weights

    return df


# Explicabilidade

# ----------------------------------------------------------------------
# Summary plot
# ----------------------------------------------------------------------

# Defina o tipo de dados esperado para os valores SHAP (pode ser um array NumPy ou uma lista de arrays)
ShapValuesType = Union[np.ndarray, List[np.ndarray], List[Any]]


def plot_shap_summary(
    shap_values: ShapValuesType,
    X_test_scaled: np.ndarray,
    selected_feature_names: List[str],
    title: str,
    save_filename: str,
    shap_values_plot: np.ndarray,
    dpi: int = 300,
):
    """
    Plota o gráfico SHAP Summary Plot para importância global das features.

    Argumentos:
    - shap_values (ShapValuesType): Valores SHAP brutos. Pode ser um array NumPy
      ou uma lista de arrays (neste caso, o primeiro elemento será usado).
    - X_test_scaled (np.ndarray): O conjunto de dados de teste (ou dados
      utilizados para gerar os valores SHAP).
    - selected_feature_names (List[str]): Nomes das features correspondentes
      às colunas de X_test_scaled.
    - title (str): Título a ser exibido no gráfico.
    - save_filename (str): Caminho e nome do arquivo para salvar o gráfico.
      Ex: '../data/08_reporting/modelo_a_shap_summary.png'
    - dpi (int): Resolução (Dots Per Inch) para salvar a imagem. Padrão é 300.
    """

    try:
        # Cria a figura e os eixos do Matplotlib
        fig, ax = plt.subplots()

        shap.summary_plot(
            shap_values_plot,
            X_test_scaled,
            feature_names=selected_feature_names,
            show=False,  # Não exibe imediatamente, controlaremos isso
        )

        plt.title(title)

        # 3. Exibição e Salvamento
        plt.show()  # Exibe o gráfico na interface (notebook/IDE)

        # Salva a imagem.
        plt.savefig(
            save_filename,
            dpi=dpi,
            bbox_inches="tight",  # Garante que todos os rótulos e eixos sejam incluídos
        )

    except Exception as e:
        print(f"Erro ao gerar ou salvar o gráfico SHAP: {e}")

    finally:
        # 4. Limpeza
        # Limpa o Matplotlib Figure e fecha para liberar memória
        plt.clf()
        plt.close()


# ----------------------------------------------------------------------
# Force Plot
# ----------------------------------------------------------------------

# Defina os tipos de dados
XTestType = Union[np.ndarray, pd.DataFrame]
ShapValuesPlotType = np.ndarray

# ----------------------------------------------------------------------
# 1. Função para Extrair e Formatar os Dados da Observação
# ----------------------------------------------------------------------


def get_observation_data(
    X_test_scaled: XTestType,
    shap_values_plot: ShapValuesPlotType,
    observation_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai e formata os dados de uma observação específica.

    Retorna:
    - X_obs_values (np.ndarray): Valores da feature, arredondados para 2 casas.
    - X_obs_for_prediction (np.ndarray): A observação em formato 2D para predição.
    - shap_obs_values (np.ndarray): Valores SHAP 1D para a observação.
    """
    # Acessar os valores da feature (a linha de dados)
    if hasattr(X_test_scaled, "iloc"):  # Pandas DataFrame/Series
        X_obs_values_raw = X_test_scaled.iloc[observation_index].values
    else:  # NumPy Array
        X_obs_values_raw = X_test_scaled[observation_index]

    # ARREDONDAR PARA DUAS CASAS DECIMAIS para exibição no Force Plot
    X_obs_values = np.round(X_obs_values_raw, 2)

    # Para predição, o modelo sempre espera uma entrada 2D
    X_obs_for_prediction = X_test_scaled[observation_index : observation_index + 1]

    # Valores SHAP para a observação
    shap_obs_values = shap_values_plot[observation_index]

    return X_obs_values, X_obs_for_prediction, shap_obs_values


# ----------------------------------------------------------------------
# 2. Função para Fazer a Predição e Imprimir Informações
# ----------------------------------------------------------------------


def print_prediction_info(
    model: Any,
    X_obs_for_prediction: np.ndarray,
    explainer_expected_value: Union[float, np.ndarray],
    observation_index: int,
) -> float:
    """
    Faz a predição, imprime o valor base e o valor previsto.

    Retorna:
    - predicted_value_float (float): O valor previsto como um float.
    """
    # 1. Fazer a predição
    predicted_value_raw = model.predict(X_obs_for_prediction)

    # 2. Simplificar o valor previsto para um float único
    if isinstance(predicted_value_raw, list):
        # Trata lista de arrays (ex: saídas de Keras multi-output)
        predicted_value = predicted_value_raw[0].flatten()[0]
    elif predicted_value_raw.ndim > 1:
        # Trata array 2D (ex: shape (1, 1))
        predicted_value = predicted_value_raw.flatten()[0]
    else:
        # Trata array 1D ou valor único
        predicted_value = predicted_value_raw.flatten()[0]

    predicted_value_float = float(predicted_value)

    # 3. Simplificar o valor esperado para um float único
    if isinstance(explainer_expected_value, np.ndarray):
        expected_value_float = float(explainer_expected_value.flatten()[0])
    else:
        expected_value_float = float(explainer_expected_value)

    # 4. Imprimir
    print(f"\n--- Explicação SHAP Force Plot para a Observação {observation_index} ---")
    print(f"Valor Base (Média): {expected_value_float:.4f}")
    print(f"Valor Previsto: {predicted_value_float:.4f}")
    print("-" * 50)

    return predicted_value_float


# ----------------------------------------------------------------------
# 3. Função para Plotar e Salvar o Force Plot
# ----------------------------------------------------------------------


def plot_and_save_force(
    explainer_expected_value: Union[float, np.ndarray],
    shap_obs_values: np.ndarray,
    X_obs_values: np.ndarray,
    selected_feature_names: List[str],
    observation_index: int,
    save_filename_prefix: str,
    dpi: int,
):
    """
    Gera o gráfico SHAP Force Plot e salva o resultado.
    """
    try:
        # Geração do Force Plot
        shap.force_plot(
            base_value=explainer_expected_value,
            shap_values=shap_obs_values,
            features=X_obs_values,
            feature_names=selected_feature_names,
            matplotlib=True,  # Força o uso de Matplotlib para salvar
            show=False,
        )

        # Salvar a figura atual do Matplotlib
        final_save_filename = f"{save_filename_prefix}_obs_{observation_index}.png"
        plt.savefig(final_save_filename, dpi=dpi, bbox_inches="tight")
        print(f"Force Plot salvo em: {final_save_filename}")

        plt.show()  # Exibe o gráfico (opcional)

    except Exception as e:
        print(f"ERRO: Não foi possível gerar ou salvar o Force Plot: {e}")

    finally:
        plt.clf()
        plt.close()


def generate_shap_force_plot(
    model: Any,
    explainer_expected_value: Union[float, np.ndarray],
    shap_values_plot: ShapValuesPlotType,
    X_test_scaled: XTestType,
    selected_feature_names: List[str],
    observation_index: int,
    save_filename_prefix: str = "../data/08_reporting/shap_force_plot",
    dpi: int = 300,
):
    """
    Coordena a extração de dados, a predição e a plotagem do SHAP Force Plot.
    """
    try:
        # 1. Extrair e formatar os dados da observação
        X_obs_values, X_obs_for_prediction, shap_obs_values = get_observation_data(
            X_test_scaled, shap_values_plot, observation_index
        )

        # 2. Fazer a predição e imprimir informações
        _ = print_prediction_info(
            model, X_obs_for_prediction, explainer_expected_value, observation_index
        )

        # 3. Plotar e salvar o gráfico
        plot_and_save_force(
            explainer_expected_value,
            shap_obs_values,
            X_obs_values,
            selected_feature_names,
            observation_index,
            save_filename_prefix,
            dpi,
        )

    except Exception as e:
        print(
            f"\nERRO FATAL na geração do SHAP Force Plot para a observação {observation_index}: {e}"
        )


# ----------------------------------------------------------------------
# Force plot
# ----------------------------------------------------------------------
