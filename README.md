
# Proteção de Dados de Clientes com Machine Learning

## Sobre o projeto

Este projeto foi desenvolvido para a seguradora Proteja Seu Amanhã com o objetivo de explorar aplicações de Machine Learning na análise de dados de clientes, incluindo identificação de clientes semelhantes, previsão de recebimento de benefícios de seguro, estimativa de valores e proteção de informações pessoais por meio de técnicas de ofuscação de dados.

O trabalho utiliza algoritmos de vizinhos mais próximos (KNN), Regressão Linear e transformações matriciais para investigar como os dados podem ser utilizados em modelos preditivos sem comprometer a qualidade das previsões.

Projeto desenvolvido como parte da formação em Data Science da TripleTen.

## Objetivos

- Identificar clientes semelhantes utilizando o algoritmo KNN.
- Avaliar o impacto do escalonamento das variáveis nas métricas de distância.
- Construir um modelo de classificação para prever o recebimento de benefícios de seguro.
- Comparar o KNN com um modelo de referência aleatório (*Dummy Classifier*).
- Implementar um modelo de Regressão Linear utilizando Álgebra Linear.
- Avaliar o desempenho da regressão com a métrica REQM (RMSE).
- Ofuscar as características dos clientes por meio de uma matriz invertível.
- Verificar se a ofuscação altera as previsões e o desempenho da Regressão Linear.

## Tecnologias e ferramentas

- **Python** — desenvolvimento e análise de dados.
- **Pandas** — manipulação de dados.
- **NumPy** — operações matriciais e cálculos numéricos.
- **Scikit-learn** — implementação e avaliação dos modelos de Machine Learning.
- **Seaborn** — visualização e análise exploratória.
- **Matplotlib** — suporte à visualização de dados.
- **Jupyter Notebook** — desenvolvimento e documentação das análises.

## Conjunto de dados

O conjunto de dados contém informações de 5.000 clientes da seguradora, com as seguintes variáveis:

| Variável | Descrição |
|---|---|
| `gender` | Gênero do cliente, codificado numericamente |
| `age` | Idade do cliente |
| `income` | Renda do cliente |
| `family_members` | Quantidade de integrantes da família |
| `insurance_benefits` | Quantidade de benefícios de seguro recebidos |

Durante o pré-processamento, a coluna de idade foi convertida para o tipo inteiro. Também foi criada a variável binária `insurance_benefits_received`, indicando se o cliente recebeu algum benefício de seguro.

A análise da variável de classificação identificou um desequilíbrio entre as classes:

- **4.436 clientes (88,72%)** não receberam benefícios.
- **564 clientes (11,28%)** receberam algum benefício.

Esse desequilíbrio foi considerado na avaliação dos modelos de classificação.

## Etapas do projeto

### 1. Análise exploratória dos dados

Foram realizadas verificações iniciais da estrutura dos dados, dos tipos das variáveis e das estatísticas descritivas. Também foi utilizado um gráfico de pares (*pairplot*) para explorar as relações entre as características dos clientes.

A análise mostrou que as variáveis possuem escalas diferentes, especialmente a renda. Essa diferença é relevante para algoritmos baseados em distância, como o KNN, pois variáveis com valores numericamente maiores podem dominar o cálculo das distâncias.

### 2. Identificação de clientes semelhantes

Foi implementada uma função utilizando `NearestNeighbors` para recuperar os cinco clientes mais próximos de um registro selecionado.

Foram comparadas quatro combinações:

- Dados originais com distância Euclidiana.
- Dados escalados com distância Euclidiana.
- Dados originais com distância Manhattan.
- Dados escalados com distância Manhattan.

O escalonamento foi realizado com `MaxAbsScaler`.

**Resultado:** os dados escalados permitiram comparar as características em escalas mais equilibradas, reduzindo a influência desproporcional da renda no cálculo das distâncias.

Essa etapa demonstrou a importância do pré-processamento para a qualidade da análise de similaridade.

### 3. Classificação do recebimento de benefícios

O problema foi formulado como uma classificação binária:

- `0`: cliente não recebeu benefícios.
- `1`: cliente recebeu algum benefício.

Foi utilizado o algoritmo KNN, testando valores de `k` entre 1 e 10, tanto com os dados originais quanto com os dados escalados.

A métrica escolhida foi o F1-score, que combina precisão (*precision*) e revocação (*recall*), sendo útil em problemas com classes desbalanceadas.

#### Resultados da classificação

O melhor resultado foi obtido com os dados escalados e `k = 5`.

| Modelo | F1-score |
|---|---:|
| Dummy aleatório — melhor resultado observado | 0,19 |
| KNN com dados originais — `k = 1` | 0,65 |
| KNN com dados escalados — `k = 1` | 0,93 |
| KNN com dados escalados — `k = 3` | 0,94 |
| KNN com dados escalados — `k = 5` | **0,95** |

O KNN com dados escalados apresentou um desempenho significativamente superior ao modelo de referência. O resultado evidencia o impacto do escalonamento na classificação por vizinhos mais próximos.

### 4. Regressão Linear para estimativa dos benefícios

Foi implementada uma classe própria, `MyLinearRegression`, utilizando a equação normal da Regressão Linear:

\[
w = (X^T X)^{-1}X^T y
\]

O modelo foi treinado para estimar a quantidade de benefícios de seguro a partir das características dos clientes.

Os dados foram divididos em 70% para treinamento e 30% para teste. O desempenho foi avaliado com a métrica REQM (*Root Mean Squared Error*).

#### Resultados da regressão

| Configuração | REQM |
|---|---:|
| Dados originais | 0,34 |
| Dados escalados com `MaxAbsScaler` | 0,34 |

O REQM foi de aproximadamente **0,3436** nas duas configurações.

Nesse experimento, o escalonamento não alterou o erro observado da Regressão Linear. Isso é consistente com a invariância das previsões da solução linear exata diante de uma transformação linear invertível das características, desde que o modelo e os dados sejam tratados de forma consistente.

### 5. Ofuscação dos dados

Para investigar a proteção das informações pessoais, foi aplicada uma transformação matricial às características dos clientes:

\[
X' = XP
\]

Em que:

- \(X\) representa a matriz original de características.
- \(P\) representa uma matriz quadrada invertível.
- \(X'\) representa a matriz de características ofuscadas.

A matriz foi gerada com NumPy e sua invertibilidade foi verificada por meio do determinante.

O determinante obtido foi aproximadamente **0,24339136**, confirmando que a matriz utilizada era invertível.

A recuperação dos dados originais foi realizada utilizando a matriz inversa:

\[
X = X'P^{-1}
\]

Os valores recuperados apresentaram diferenças numéricas extremamente pequenas em relação aos originais, decorrentes da precisão finita dos cálculos computacionais.

### 6. Validação da ofuscação com Regressão Linear

Após a transformação dos dados, foi treinado um modelo de Regressão Linear com as características ofuscadas. O resultado foi comparado ao modelo treinado com os dados originais.

| Configuração | REQM |
|---|---:|
| Modelo com dados originais | 0,3435565089 |
| Modelo com dados ofuscados | 0,3435565089 |

O erro permaneceu exatamente igual na precisão apresentada pelo notebook.

Esse resultado confirma experimentalmente que, neste cenário, a transformação matricial utilizada não prejudicou a qualidade preditiva da Regressão Linear.

A demonstração matemática apresentada no projeto também mostra que, para uma matriz de transformação invertível, os pesos podem mudar enquanto as previsões permanecem equivalentes.

**Observação:** a ofuscação é uma transformação dos dados, não uma garantia completa de anonimização ou segurança. Em aplicações reais, a proteção de informações pessoais exige medidas adicionais de segurança e governança.

## Principais resultados

- **F1-score de 0,95** com KNN e dados escalados, utilizando `k = 5`.
- Desempenho do KNN escalado superior ao do modelo de referência aleatório.
- REQM de aproximadamente **0,3436** na Regressão Linear, tanto com dados originais quanto escalados.
- Construção e validação de uma matriz invertível para ofuscação das características.
- Recuperação dos dados originais a partir da transformação matricial, com pequenas diferenças numéricas.
- Manutenção do REQM em **0,3435565089** após a ofuscação dos dados.
- Demonstração matemática e experimental da preservação das previsões da Regressão Linear sob a transformação utilizada.

## Conclusão

O projeto demonstrou diferentes aplicações de Machine Learning no contexto de uma seguradora, desde a identificação de clientes semelhantes até a classificação e a estimativa de benefícios de seguro.

Na tarefa de classificação, o KNN com dados escalados alcançou F1-score de 0,95, mostrando que o tratamento adequado das escalas das variáveis pode melhorar significativamente o desempenho de algoritmos baseados em distância.

Na tarefa de regressão, o modelo apresentou REQM de aproximadamente 0,3436. Esse resultado foi mantido após a ofuscação das características por uma matriz invertível, indicando que a transformação escolhida preservou a qualidade das previsões nesse experimento.

Dessa forma, o projeto evidencia como técnicas de análise de dados, Machine Learning e Álgebra Linear podem ser combinadas para desenvolver soluções preditivas e investigar estratégias de proteção de informações pessoais.

## Competências desenvolvidas

- Análise exploratória e preparação de dados.
- Manipulação de dados com Pandas e NumPy.
- Identificação de clientes semelhantes com KNN.
- Classificação binária e avaliação por F1-score.
- Comparação com modelos de referência (*Dummy*).
- Escalonamento de características com `MaxAbsScaler`.
- Implementação de Regressão Linear com Álgebra Linear.
- Avaliação de modelos com REQM.
- Operações matriciais, determinantes e matrizes inversas.
- Ofuscação e recuperação de dados.
- Validação experimental e interpretação de resultados.

