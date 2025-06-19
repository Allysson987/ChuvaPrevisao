# 🌧️ Previsão de Chuvas com Dados do INMET

Este projeto utiliza dados meteorológicos públicos do **INMET - Estação de Caruaru (PE)** para treinar e testar diversos modelos de **regressão supervisionada** com o objetivo de prever o índice de chuvas.

---

## 📌 Objetivo

O objetivo principal deste código é **definir os melhores padrões de análise** e **testar modelos preditivos** de chuvas com base em variáveis meteorológicas. Posteriormente, os modelos serão ajustados para **trabalhar com os dados reais capturados por sensores físicos** de uma **estação meteorológica escolar**, localizada em **Jaboatão dos Guararapes (PE)**, via API própria.

---

## 📊 Dados Utilizados

- **Fonte:** [INMET - Instituto Nacional de Meteorologia](https://bdmep.inmet.gov.br)
- **Local:** Caruaru - PE
- **Variáveis utilizadas no modelo:**
  - Umidade mínima
  - Umidade máxima
  - Ponto de orvalho mínimo

> ⚠️ **Observação:** Mesmo com mais variáveis disponíveis nos dados originais, este projeto utilizou **apenas três variáveis climáticas** para avaliar a capacidade preditiva com entradas limitadas. O modelo será futuramente ajustado com os sensores físicos da estação meteorológica.

---

## 🤖 Modelos Testados

Foram testados e comparados os seguintes modelos de regressão:

- Árvores de Decisão (Decision Tree Regressor)
- Floresta Aleatória (Random Forest Regressor)
- Gradient Boosting
- AdaBoost
- KNN Regressor
- SVR (Support Vector Regressor)

### 🔍 Métricas de avaliação:
- MAE (Erro Absoluto Médio)
- MSE (Erro Quadrático Médio)
- RMSE (Raiz do Erro Quadrático Médio)
- R² (Coeficiente de Determinação)

Modelos com desempenho satisfatório (**R² ≥ 0.35**) são salvos em arquivos `.joblib` para uso futuro com dados reais.

---

## 🔄 Integração com Estação Meteorológica

Este projeto será integrado com uma **estação meteorológica física** em desenvolvimento, que coleta dados em tempo real via sensores e envia para uma API. A integração ocorrerá com o repositório:

📡 **Estação Meteorológica com IoT:**  
🔗 [github.com/Allysson987/Monitorameno-de-temperatura](https://github.com/Allysson987/Monitorameno-de-temperatura/tree/main)

---

## 📞 Contato

**Desenvolvedor:** Allysson Silva Pereira  
📧 silvapereiraallysson51@gmail.com  
📱 (81) 9 9723-3867

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- scikit-learn
- NumPy
- joblib

---

## 📁 Estrutura do Código

- `Analise()`: Classe principal que carrega os dados, realiza o pré-processamento, treina os modelos e faz previsões.
- `preparacao()`: Limpeza e organização dos dados meteorológicos.
- `treinar()`: Treinamento do modelo com agrupamento diário.
- `modelos()`: Execução e avaliação dos modelos.
- `predir()`: Realiza previsões futuras com base em novos dados.

---

## 📌 Observação Final

Este projeto é um protótipo para aplicações em **educação, pesquisa e integração com IoT**, com foco em **monitoramento climático inteligente e acessível**.

---
