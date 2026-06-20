# Resumo Técnico: Modelo Preditivo Copa do Mundo 2026

Este documento serve como um guia de bolso técnico para as metodologias matemáticas e computacionais implementadas no notebook `seminario_copa_2026.ipynb`. Ideal para responder a perguntas aprofundadas durante o seminário.

## 1. Aquisição e Preparação de Dados (Pipeline ETL)
- **Fonte de Dados:** Dataset histórico da FIFA via Kaggle (`results.csv`).
- **Filtragem Lógica:** Utiliza-se `pandas` para truncar o dataset a partir de meados de 2022, isolando o ciclo atual da Copa do Mundo. Apenas torneios oficiais de interesse (`Friendly` e `FIFA World Cup`) são mantidos.
- **Abstração Neutra:** Diferente de campeonatos de pontos corridos, o modelo foca em predições de **campo neutro**, eliminando o viés do Fator Casa (*Home Advantage*) tradicionalmente usado em modelos matemáticos esportivos.

## 2. Modelo Base: Poisson Independente (Ato 1)
O ponto de partida é a modelagem de gols como **variáveis aleatórias de Poisson independentes**.
- **A Lógica:** A taxa de gols esperados de uma seleção $i$ contra uma seleção $j$ ($\lambda_i$) é puramente o produto do poder de ataque de $i$ pelo poder de defesa de $j$.
- **Fórmula:** $P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}$
- **Vulnerabilidades:** Assume independência absoluta entre os gols do Time A e Time B. Na realidade esportiva, a ocorrência de um gol afeta o comportamento defensivo/ofensivo subsequente. O modelo de Poisson puro notoriamente subestima placares de empate baixo (0x0, 1x1).

## 3. A Evolução: Modelo Dixon-Coles (Ato 2)
Implementação do *paper* clássico de Dixon e Coles (1997), com adaptações modernas de Machine Learning.

### 3.1 Correção de Dependência Bivariada ($\tau$)
Introduz-se o fator de correção $\tau$ (Rho) na função de probabilidade conjunta.
- **Mecânica:** O parâmetro inflaciona artificialmente as probabilidades na vizinhança do 0x0 e 1x1, e deflaciona marginalmente o 1x0 e 0x1, "puxando" a distribuição probabilística para o centro (empates) para refletir a interdependência defensiva no futebol real.

### 3.2 Decaimento de Tempo (Time Decay)
O futebol é dinâmico; o momento (*form*) importa.
- **Mecânica:** Cada partida histórica recebe um peso probabilístico $e^{-\xi t}$, onde $t$ é o número de dias desde a partida e $\xi$ é a constante de decaimento (meia-vida). Jogos mais recentes têm impacto massivo na estimativa dos parâmetros; jogos de 4 anos atrás têm seu impacto diluído a quase zero.

### 3.3 Otimização Numérica (Máxima Verossimilhança)
Este é o núcleo computacional pesado do notebook.
- **O Algoritmo:** Utilizamos `scipy.optimize.minimize` (com o método `SLSQP` ou `L-BFGS-B`).
- **A Restrição (Constraint):** Para evitar que os parâmetros fujam para o infinito (já que ataque e defesa são relativos), forçamos matematicamente que a soma dos parâmetros de ataque de todas as seleções seja igual a zero ($\sum \alpha = 0$).
- **Vetorização (Performance):** A Função de Log-Verossimilhança (Log-Likelihood) é processada simultaneamente para toda a matriz histórica utilizando arrays do `numpy` compilados em C, eliminando laços `for` lentos do Python. Isso viabiliza que o modelo otimize centenas de parâmetros ($\alpha$, $\beta$, $\tau$) em poucos segundos no processador (CPU).

## 4. Inferência e Valor Esperado (EV)
Transformação das predições estocásticas em informações aplicáveis a mercados financeiros (Bolsas de Apostas Esportivas).
- **Extração Geométrica:** A matriz de probabilidade bivariada (Ex: 6x6) é desconstruída utilizando operações geométricas (`numpy.tril`, `numpy.triu`, `numpy.diag`) para consolidar as chances em: Probabilidade de Vitória do Time A, Empate, e Vitória do Time B.
- **Cálculo da Odd Justa (Fair Odd):** $1 / P(Evento)$.
- **Valor Esperado (EV):** O modelo procura discrepâncias entre a probabilidade implícita do mercado e a "Odd Justa" extraída da matriz Dixon-Coles rigorosa para atestar se existe EV positivo (+EV) no evento estatístico.
