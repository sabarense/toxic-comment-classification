import pandas as pd
import os
from tqdm import tqdm

# Importações para plotagem
import matplotlib.pyplot as plt
import seaborn as sns

from src.text_preprocessing import preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

# scikit-multilearn
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB # Se quiser testar outro classificador


# --- 2. Carregar os Dados ---
print("Carregando dados...")

data_path = 'data'
train_file = os.path.join(data_path, 'train.csv')
test_file = os.path.join(data_path, 'test.csv')

try:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivos não encontrados. Verifique se '{train_file}' e '{test_file}' existem.")
    print("Você pode baixar os dados em: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
    exit() # Em um script, é comum sair se dados críticos não forem encontrados.

# Visualizar as primeiras linhas
print("\n--- Amostra dos Dados de Treino ---")
print(train_df.head())
print(f"\nShape do treino: {train_df.shape}")
print(f"Shape do teste: {test_df.shape}")

# Definir as colunas de rótulos (categorias de toxicidade)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Verificar a distribuição dos rótulos
print("\n--- Distribuição dos Rótulos (Treino) ---")
print(train_df[labels].sum())

# Verificar valores nulos (se houver, preencher com string vazia)
# CORREÇÃO: Usar atribuição direta para evitar FutureWarning do Pandas
train_df['comment_text'] = train_df['comment_text'].fillna('')
test_df['comment_text'] = test_df['comment_text'].fillna('')


# --- 3. Pré-processamento de Texto ---

print("\n--- Pré-processando textos de treino---")
# Usar list comprehension com tqdm para barras de progresso em scripts
train_df['processed_comment_text'] = [
    preprocess_text(text) for text in tqdm(train_df['comment_text'], desc="Pré-processando treino")
]


print("\n--- Pré-processando textos de teste---")
test_df['processed_comment_text'] = [
    preprocess_text(text) for text in tqdm(test_df['comment_text'], desc="Pré-processando teste")
]

print("\nPré-processamento concluído. Amostra do texto processado:")
print(train_df[['comment_text', 'processed_comment_text']].head())


# --- 4. Vetorização de Texto (TF-IDF) ---

print("\n--- Vetorizando textos com TF-IDF... ---")
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2))

X_train = tfidf_vectorizer.fit_transform(train_df['processed_comment_text'])
X_test = tfidf_vectorizer.transform(test_df['processed_comment_text'])
Y_train = train_df[labels].values

print(f"Shape de X_train (vetorizado): {X_train.shape}")
print(f"Shape de X_test (vetorizado): {X_test.shape}")
print(f"Shape de Y_train (rótulos): {Y_train.shape}")


# --- 5. Modelagem e Treinamento (Classificação Multilabel) ---

print("\n--- Treinando Modelos Multilabel ---")

# 5.1. Binary Relevance
print("\n--- Treinando Binary Relevance com LinearSVC ---")
classifier_br = BinaryRelevance(classifier=LinearSVC(), require_dense=[False, True])
classifier_br.fit(X_train, Y_train)
predictions_br = classifier_br.predict(X_test)
print("Treinamento Binary Relevance concluído.")

# 5.2. Classifier Chains
print("\n--- Treinando Classifier Chains com LinearSVC ---")
classifier_cc = ClassifierChain(classifier=LinearSVC(), require_dense=[False, True])
classifier_cc.fit(X_train, Y_train)
predictions_cc = classifier_cc.predict(X_test)
print("Treinamento Classifier Chains concluído.")

# 5.3. Label Powerset
print("\n--- Treinando Label Powerset com LinearSVC ---")
classifier_lp = LabelPowerset(classifier=LinearSVC(), require_dense=[False, True])
classifier_lp.fit(X_train, Y_train)
predictions_lp = classifier_lp.predict(X_test)
print("Treinamento Label Powerset concluído.")


# --- 6. Avaliação do Desempenho (com divisão Treino/Validação SIMULADA) ---

print("\n--- Avaliando o Desempenho dos Modelos (Simulação Treino/Validação) ---")

X_train_eval, X_val_eval, Y_train_eval, Y_val_eval = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)

print("Re-treinando Binary Relevance para avaliação...")
classifier_br_eval = BinaryRelevance(classifier=LinearSVC(), require_dense=[False, True])
classifier_br_eval.fit(X_train_eval, Y_train_eval)
predictions_br_eval = classifier_br_eval.predict(X_val_eval)

print("Re-treinando Classifier Chains para avaliação...")
classifier_cc_eval = ClassifierChain(classifier=LinearSVC(), require_dense=[False, True])
classifier_cc_eval.fit(X_train_eval, Y_train_eval)
predictions_cc_eval = classifier_cc_eval.predict(X_val_eval)

print("Re-treinando Label Powerset para avaliação...")
# CORREÇÃO: Garantir que Y_train_eval é usado para o fit do Label Powerset
classifier_lp_eval = LabelPowerset(classifier=LinearSVC(), require_dense=[False, True])
classifier_lp_eval.fit(X_train_eval, Y_train_eval)
predictions_lp_eval = classifier_lp_eval.predict(X_val_eval)

# Função para calcular e imprimir métricas (retorna o dicionário de métricas)
def evaluate_model(y_true, y_pred, model_name):
    """Calcula e imprime métricas de avaliação para classificação multilabel."""
    metrics = {
        "Acurácia (subconjunto exato)": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Jaccard Score (samples)": jaccard_score(y_true, y_pred, average='samples'),
        "F1-Score (samples)": f1_score(y_true, y_pred, average='samples'),
        "F1-Score (micro)": f1_score(y_true, y_pred, average='micro'),
        "F1-Score (macro)": f1_score(y_true, y_pred, average='macro')
    }
    print(f"\n--- Métricas para: {model_name} ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    return metrics # Retornar as métricas para uso em gráficos

# Dicionário para armazenar os resultados para plotagem
results = {}

# Avaliar os modelos com o conjunto de validação simulado e armazenar os resultados
results["Binary Relevance"] = evaluate_model(Y_val_eval, predictions_br_eval, "Binary Relevance (LinearSVC)")
results["Classifier Chains"] = evaluate_model(Y_val_eval, predictions_cc_eval, "Classifier Chains (LinearSVC)")
results["Label Powerset"] = evaluate_model(Y_val_eval, predictions_lp_eval, "Label Powerset (LinearSVC)")


# --- 7. Plotagem dos Resultados ---
print("\n--- Gerando Gráficos de Resultados ---")

# Converter resultados para DataFrame para facilitar a plotagem
results_df = pd.DataFrame.from_dict(results, orient='index')

# Gráfico 1: Comparação de Métricas Chave (Acurácia, Jaccard, F1 Samples)
plt.figure(figsize=(12, 7))
sns.lineplot(data=results_df[['Acurácia (subconjunto exato)', 'Jaccard Score (samples)', 'F1-Score (samples)']].T)
plt.title('Comparação de Desempenho por Modelo')
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Modelo')
plt.tight_layout()
plt.savefig('metricas_comparacao.png')
print("Gráfico 'metricas_comparacao.png' gerado.")


# Gráfico 2: F1-Score (Micro e Macro) por Modelo (Barras)
plt.figure(figsize=(10, 6))
results_df[['F1-Score (micro)', 'F1-Score (macro)']].plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('F1-Score (Micro e Macro) por Modelo')
plt.xlabel('Modelo')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Tipo de F1-Score')
plt.tight_layout()
plt.savefig('f1_scores_bar.png')
print("Gráfico 'f1_scores_bar.png' gerado.")


# Gráfico 3: Hamming Loss (quanto menor, melhor)
plt.figure(figsize=(8, 5))
results_df['Hamming Loss'].plot(kind='bar', figsize=(8, 5), rot=0, color='salmon')
plt.title('Hamming Loss por Modelo (Menor é Melhor)')
plt.xlabel('Modelo')
plt.ylabel('Hamming Loss')
plt.ylim(0, 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('hamming_loss_bar.png')
print("Gráfico 'hamming_loss_bar.png' gerado.")


print("\n--- Processo Concluído ---")