import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Carregar o dataset
df = pd.read_csv('dataset.csv')

# Exibir as primeiras linhas para inspeção
print(df.head())

# Verificar valores ausentes
print(df.isnull().sum())

# Tratamento
df = df.dropna()

df['Idade'].fillna(df['Idade'].mean(), inplace=True)
df['Renda Anual (em $)'].fillna(df['Renda Anual (em $)'].mean(), inplace=True)
df['Tempo no Site (min)'].fillna(df['Tempo no Site (min)'].mean(), inplace=True)

# Preenchendo com moda
df['Gênero'].fillna(df['Gênero'].mode()[0], inplace=True)
df['Anúncio Clicado'].fillna(df['Anúncio Clicado'].mode()[0], inplace=True)

# Verificar valores novamente
print(df.isnull().sum())

# Visualização das distribuições
sns.histplot(df['Idade'], kde=True)
plt.title('Distribuição de Idade')
plt.show()

sns.histplot(df['Renda Anual (em $)'], kde=True)
plt.title('Distribuição de Renda Anual')
plt.show()

sns.histplot(df['Tempo no Site (min)'], kde=True)
plt.title('Distribuição de Tempo no Site')
plt.show()

sns.countplot(x='Gênero', data=df)
plt.title('Distribuição de Gênero')
plt.show()

sns.countplot(x='Anúncio Clicado', data=df)
plt.title('Distribuição de Anúncio Clicado')
plt.show()

# Boxplots
sns.boxplot(x='Compra (0 ou 1)', y='Idade', data=df)
plt.title('Idade vs Compra')
plt.show()

sns.boxplot(x='Compra (0 ou 1)', y='Renda Anual (em $)', data=df)
plt.title('Renda Anual vs Compra')
plt.show()

sns.boxplot(x='Compra (0 ou 1)', y='Tempo no Site (min)', data=df)
plt.title('Tempo no Site vs Compra')
plt.show()

# Pré-processamento
numerical_features = ['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

encoder = LabelEncoder()
df['Gênero'] = encoder.fit_transform(df['Gênero'])
df['Anúncio Clicado'] = encoder.fit_transform(df['Anúncio Clicado'])

X = df[['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)', 'Gênero', 'Anúncio Clicado']]
y = df['Compra (0 ou 1)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de Classificação
models = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Regressão Logística": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Função para treinar e avaliar os modelos
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
    return results

# Avaliar os modelos
results = evaluate_models(models, X_train, X_test, y_train, y_test)

# Converter resultados em um DataFrame para visualização
results_df = pd.DataFrame(results).T
print(results_df)

# Visualizar as métricas de desempenho dos modelos
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparação de Desempenho dos Modelos de Classificação')
plt.ylabel('Métrica')
plt.show()

# Validação cruzada
def cross_validate_models(models, X, y):
    cv_results = {}
    for name, model in models.items():
        cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = cv_score.mean()
    return cv_results

# Realizar a validação cruzada
cv_results = cross_validate_models(models, X, y)
print("Validação Cruzada (Acurácia Média):")
print(cv_results)

# Visualizar os resultados da validação cruzada
cv_results_df = pd.DataFrame(cv_results, index=['Acurácia Média'])
cv_results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Validação Cruzada: Acurácia Média por Modelo')
plt.ylabel('Acurácia Média')
plt.show()

# Ajuste de Hiperparâmetros com GridSearchCV para o RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Melhor modelo e seus parâmetros
print(f"Melhor modelo: {grid_search.best_estimator_}")
print(f"Melhores parâmetros: {grid_search.best_params_}")

# Avaliar o desempenho do melhor modelo
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Acurácia do Melhor Modelo: {accuracy_best:.4f}')
