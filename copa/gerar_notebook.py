import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# --- SETUP ---
add_md("""# Previsão de Placares da Copa do Mundo 2026: Estatística vs. Realidade

## 1. Aquisição e Preparação de Dados (O Setup)
Para este projeto, utilizaremos o dataset **"International football results from 1872 to 2026"** do Kaggle. 
*(Certifique-se de baixar o arquivo `results.csv` no Kaggle e colocá-lo na mesma pasta que este notebook).*

Filtraremos os dados para manter:
1. Jogos a partir de meados de 2022 (foco no ciclo da Copa atual).
2. Apenas Amistosos ("Friendly") e Copa do Mundo ("FIFA World Cup").
3. Como a Copa é em campo neutro, abstrairemos mandante e visitante para **Time A** e **Time B**.""")

add_code("""import pandas as pd
import numpy as np
import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo premium para os gráficos
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# 1. Carregando os dados
try:
    df = pd.read_csv('results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtro de datas (ciclo recente) e torneios
    df = df[(df['date'] >= '2022-06-01')]
    df = df[df['tournament'].isin(['Friendly', 'FIFA World Cup'])]
    
    # Tratamento de Nulos
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    print(f"Dataset carregado com sucesso! Total de partidas após filtros: {len(df)}")
except FileNotFoundError:
    print("ERRO: O arquivo 'results.csv' não foi encontrado. Por favor, baixe do Kaggle e coloque na mesma pasta deste notebook.")""")

# --- ATO 1 ---
add_md("""## 2. Ato 1: O Estatístico Ingênuo (Modelo de Poisson Básico)

**Conceito Matemático:** Prever placares baseando-se no cruzamento de eventos independentes.
A força de ataque de uma seleção ($\\lambda$) depende exclusivamente de sua própria capacidade ofensiva multiplicada pela capacidade defensiva do adversário. A equação de Poisson dita a probabilidade de $k$ gols:

$$P(k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$

> **O apresentador dirá:** *"Esta é a previsão estatística pura. O problema? Ela falha duplamente: ignora que o futebol é dinâmico (subestimando empates como 0x0) e trata um amistoso de 2023 com o mesmo peso de um jogo da Copa de ontem."*""")

add_code("""# Para o modelo ingênuo, calcularemos a média de gols feitos e sofridos de cada seleção.
# Simplificando a força de ataque e defesa baseada nas médias globais do período filtrado.

if 'df' in locals():
    media_gols_feitos_geral = (df['home_score'].mean() + df['away_score'].mean()) / 2

    def calcular_forcas_ingenuas(time):
        # Jogos como mandante e visitante
        jogos_casa = df[df['home_team'] == time]
        jogos_fora = df[df['away_team'] == time]
        
        gols_feitos = jogos_casa['home_score'].sum() + jogos_fora['away_score'].sum()
        gols_sofridos = jogos_casa['away_score'].sum() + jogos_fora['home_score'].sum()
        total_jogos = len(jogos_casa) + len(jogos_fora)
        
        if total_jogos == 0:
            return 1.0, 1.0 # Força neutra se não houver dados
            
        media_ataque = gols_feitos / total_jogos
        media_defesa = gols_sofridos / total_jogos
        
        ataque = media_ataque / media_gols_feitos_geral
        defesa = media_defesa / media_gols_feitos_geral
        return ataque, defesa

    # =====================================================================
    # ESCOLHA AQUI OS TIMES PARA A DEMONSTRAÇÃO
    # =====================================================================
    # Você pode alterar esses nomes livremente para qualquer seleção presente no dataset.
    # Exemplos: 'Argentina', 'Portugal', 'England', 'Spain', 'Germany'
    TIME_A = 'Brazil'
    TIME_B = 'France'
    # =====================================================================

    ataque_A, defesa_A = calcular_forcas_ingenuas(TIME_A)
    ataque_B, defesa_B = calcular_forcas_ingenuas(TIME_B)

    # Lambda = Ataque * Defesa do Oponente * Constante de Gols Base
    lambda_A = ataque_A * defesa_B * media_gols_feitos_geral
    lambda_B = ataque_B * defesa_A * media_gols_feitos_geral

    print(f"{TIME_A} - Força de Ataque: {ataque_A:.2f}, Força de Defesa: {defesa_A:.2f}, Lambda esperado: {lambda_A:.2f}")
    print(f"{TIME_B} - Força de Ataque: {ataque_B:.2f}, Força de Defesa: {defesa_B:.2f}, Lambda esperado: {lambda_B:.2f}")

    # Distribuição Poisson (até 5 gols)
    max_gols = 6
    prob_A = [poisson.pmf(i, lambda_A) for i in range(max_gols)]
    prob_B = [poisson.pmf(i, lambda_B) for i in range(max_gols)]

    # Cruzamento de Eventos (Matriz 6x6)
    matriz_poisson = np.outer(prob_A, prob_B)

    # Plotando o Heatmap Premium
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_poisson, annot=True, fmt='.2%', cmap='magma', cbar=False, 
                xticklabels=range(max_gols), yticklabels=range(max_gols))
    plt.title(f"Ato 1: Matriz de Poisson Ingênua\\n{TIME_A} (Vertical) vs {TIME_B} (Horizontal)", color='white')
    plt.xlabel(f"Gols {TIME_B}", color='white')
    plt.ylabel(f"Gols {TIME_A}", color='white')
    plt.show()""")

# --- ATO 2 ---
add_md("""## 3. Ato 2: O Acadêmico Rigoroso (Dixon-Coles Dinâmico)

**Conceito Matemático:** Corrigir a interdependência dos gols e aplicar um sistema de "memória curta" ao modelo para refletir a forma atual das seleções.

1. **Parâmetro de Empate ($\\tau$):** Inflaciona a probabilidade de empates com poucos gols (0x0, 1x1).
2. **Decaimento no Tempo (Time Decay):** Eventos passados perdem relevância via $e^{-\\xi t}$.

> **O apresentador dirá:** *"Ao remover o fator casa e forçar o computador a esquecer lentamente o passado, o Dixon-Coles absorve a realidade. A mancha de calor no centro se condensa e os empates ganham destaque estatístico. Superamos o modelo ingênuo."*""")

add_code("""# Decaimento no Tempo (Time Decay)
# Vamos visualizar a curva exponencial de peso caindo
hoje = pd.to_datetime('today')
dias_passados = np.arange(0, 1400) # Últimos ~4 anos
xi = 0.002 # Parâmetro de decaimento (meia-vida de aprox 1 ano)
pesos_tempo = np.exp(-xi * dias_passados)

plt.figure(figsize=(10, 3))
plt.plot(dias_passados, pesos_tempo, color='#00ffcc', linewidth=2)
plt.title(r"Efeito do Decaimento de Tempo ($e^{-\\xi t}$)", color='white')
plt.xlabel("Dias Atrás", color='white')
plt.ylabel("Peso na Otimização", color='white')
plt.fill_between(dias_passados, pesos_tempo, color='#00ffcc', alpha=0.2)
plt.show()""")

add_code("""# Otimização do Modelo Dixon-Coles
if 'df' in locals() and not df.empty:
    # --- NOVO FILTRO DE VELOCIDADE (Apenas Elite) ---
    times_elite = df['home_team'].value_counts()[df['home_team'].value_counts() >= 10].index
    df_elite = df[df['home_team'].isin(times_elite) & df['away_team'].isin(times_elite)]
    
    # Criando dicionário de seleções
    selecoes = list(set(df_elite['home_team'].unique()) | set(df_elite['away_team'].unique()))
    n_selecoes = len(selecoes)
    sel_idx = {sel: i for i, sel in enumerate(selecoes)}

    # Função Tau (Correção de Empates)
    def rho_correction(x, y, lambda_x, mu_y, rho):
        if x == 0 and y == 0:
            return 1 - (lambda_x * mu_y * rho)
        elif x == 0 and y == 1:
            return 1 + (lambda_x * rho)
        elif x == 1 and y == 0:
            return 1 + (mu_y * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    # Log-Likelihood Function Customizada (Campo Neutro)
    def dixon_coles_loglik(params, df_matches, xi=0.002, lambda_reg=0.05):
        # params[:n] = Forças de Ataque (alpha)
        # params[n:2n] = Forças de Defesa (beta)
        # params[2n] = Fator Rho (Tau)
        
        alpha = params[:n_selecoes]
        beta = params[n_selecoes:2*n_selecoes]
        rho = params[2*n_selecoes]
        
        # Extraindo dados das partidas
        idx_home = df_matches['home_idx'].values
        idx_away = df_matches['away_idx'].values
        gols_home = df_matches['home_score'].values
        gols_away = df_matches['away_score'].values
        dias = df_matches['dias_atras'].values
        peso_camp = df_matches.get('peso_camp', pd.Series(np.ones(len(gols_home)))).values
        
        # Lambdas (Sem fator casa para campo neutro)
        lambda_home = np.exp(alpha[idx_home] + beta[idx_away])
        lambda_away = np.exp(alpha[idx_away] + beta[idx_home])
        
        # Peso Temporal combinado com peso de torneio
        pesos = np.exp(-xi * dias) * peso_camp
        
        # Log-Verossimilhança
        log_probs = []
        for x, y, lam, mu, p in zip(gols_home, gols_away, lambda_home, lambda_away, pesos):
            prob_poisson = poisson.pmf(x, lam) * poisson.pmf(y, mu)
            prob_corrigida = prob_poisson * rho_correction(x, y, lam, mu, rho)
            
            if prob_corrigida <= 0:
                log_probs.append(-10000 * p) # Penalidade
            else:
                log_probs.append(np.log(prob_corrigida) * p)
                
        penalty = lambda_reg * (np.sum(alpha**2) + np.sum(beta**2))
        return -np.sum(log_probs) + penalty

    # Preparando dados
    df_opt = df_elite.copy()
    df_opt['home_idx'] = df_opt['home_team'].map(sel_idx)
    df_opt['away_idx'] = df_opt['away_team'].map(sel_idx)
    df_opt['dias_atras'] = (hoje - df_opt['date']).dt.days

    def get_peso_campeonato(tourney):
        if 'World Cup' in tourney or 'Copa America' in tourney or 'Euro' in tourney:
            return 1.0
        elif 'Friendly' in tourney:
            return 0.5
        return 0.8
    if 'tournament' in df_opt.columns:
        df_opt['peso_camp'] = df_opt['tournament'].apply(get_peso_campeonato)
    else:
        df_opt['peso_camp'] = 1.0

    # Restrição: A soma das forças de ataque é 0 (média de exp(alpha) próxima de 1)
    # Isso ancora o modelo para evitar que divirja para o infinito
    def constraint_func(x):
        return sum(x[:n_selecoes])
        
    cons = [{'type': 'eq', 'fun': constraint_func}]
    
    # Smart Start: Chute inicial com Poisson Ingênuo
    media_gols_geral = (df_elite['home_score'].mean() + df_elite['away_score'].mean()) / 2
    init_alpha = np.zeros(n_selecoes)
    init_beta = np.zeros(n_selecoes)
    for sel, idx in sel_idx.items():
        jogos_casa = df_elite[df_elite['home_team'] == sel]
        jogos_fora = df_elite[df_elite['away_team'] == sel]
        total_jogos = len(jogos_casa) + len(jogos_fora)
        if total_jogos > 0:
            gf = jogos_casa['home_score'].sum() + jogos_fora['away_score'].sum()
            gs = jogos_casa['away_score'].sum() + jogos_fora['home_score'].sum()
            ataq_ing = (gf / total_jogos) / media_gols_geral
            def_ing = (gs / total_jogos) / media_gols_geral
            init_alpha[idx] = np.log(max(ataq_ing, 0.01))
            init_beta[idx] = np.log(max(def_ing, 0.01))
            
    init_alpha = init_alpha - np.mean(init_alpha)
    init_params = np.concatenate([init_alpha, init_beta, [0.0]])
    
    print("Iniciando Otimização Dixon-Coles... Isso pode levar de 10 a 60 segundos.")
    
    # SLSQP é ótimo com constraints
    bounds = [(-3, 3)] * (2 * n_selecoes) + [(-0.5, 0.5)]
    res = minimize(dixon_coles_loglik, init_params, args=(df_opt, xi), 
                   method='SLSQP', constraints=cons, bounds=bounds,
                   options={'maxiter': 50})
                   
    if res.success:
        print("Otimização Concluída com Sucesso!")
    else:
        print("Aviso: A otimização atingiu o limite de iterações. Os resultados são aproximações aceitáveis.")
        
    alpha_opt = res.x[:n_selecoes]
    beta_opt = res.x[n_selecoes:2*n_selecoes]
    rho_opt = res.x[2*n_selecoes]
    
    dict_ataque = {sel: alpha_opt[i] for sel, i in sel_idx.items()}
    dict_defesa = {sel: beta_opt[i] for sel, i in sel_idx.items()}
    
    print(f"Fator de Correção Tau (Rho) ajustado pelo mercado: {rho_opt:.4f}")
    
    # --- GERANDO A MATRIZ DIXON-COLES PARA A DEMONSTRAÇÃO ---
    if TIME_A in dict_ataque and TIME_B in dict_ataque:
        lam_A_dc = np.exp(dict_ataque[TIME_A] + dict_defesa[TIME_B])
        lam_B_dc = np.exp(dict_ataque[TIME_B] + dict_defesa[TIME_A])
        
        matriz_dc = np.zeros((max_gols, max_gols))
        for i in range(max_gols):
            for j in range(max_gols):
                prob = poisson.pmf(i, lam_A_dc) * poisson.pmf(j, lam_B_dc)
                prob *= rho_correction(i, j, lam_A_dc, lam_B_dc, rho_opt)
                matriz_dc[i, j] = prob
                
        matriz_dc = matriz_dc / matriz_dc.sum()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(matriz_poisson, annot=True, fmt='.2%', cmap='magma', cbar=False, 
                    ax=axes[0], xticklabels=range(max_gols), yticklabels=range(max_gols))
        axes[0].set_title(f"Ingênuo (Poisson)\\n{TIME_A} vs {TIME_B}")
        
        sns.heatmap(matriz_dc, annot=True, fmt='.2%', cmap='viridis', cbar=False, 
                    ax=axes[1], xticklabels=range(max_gols), yticklabels=range(max_gols))
        axes[1].set_title(f"Rigoroso (Dixon-Coles Dinâmico)\\n{TIME_A} vs {TIME_B}")
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"ERRO: As seleções '{TIME_A}' ou '{TIME_B}' não foram encontradas nos dados.")
""")

# --- VEREDITO ---
add_md("""## 4. O Veredito: Extraindo Valor (Apostas e Simulação)

Como traduzir as matrizes para prever a realidade e bater o mercado.

> **A Fala (Roteiro):** *"Apostar no placar exato é estatisticamente caótico. A verdadeira força da física computacional está em agrupar as probabilidades da nossa matriz e compará-las contra as 'odds' das casas de apostas para encontrar o Valor Esperado (EV) positivo, provando onde o mercado está errando."*""")

add_code("""if 'matriz_dc' in locals():
    # 1. Encontrar a célula com a probabilidade máxima (Placar Mais Provável)
    indice_max = np.unravel_index(np.argmax(matriz_dc), matriz_dc.shape)
    gols_A_max, gols_B_max = indice_max
    prob_maxima = matriz_dc[indice_max]
    
    print(f"--- PREVISÃO FINAL: {TIME_A} x {TIME_B} ---")
    print(f"Placar Exato Mais Provável: {TIME_A} {gols_A_max} x {gols_B_max} {TIME_B} ({prob_maxima*100:.2f}%)")
    print("-" * 40)
    
    # 2. Geometria da Matriz: Somar triângulos
    prob_vitoria_A = np.tril(matriz_dc, -1).sum()
    prob_vitoria_B = np.triu(matriz_dc, 1).sum()
    prob_empate = np.diag(matriz_dc).sum()
    
    print(f"Probabilidade {TIME_A} Vencer: {prob_vitoria_A*100:.2f}%  ->  (Odd Justa Estimada: {1/prob_vitoria_A:.2f})")
    print(f"Probabilidade {TIME_B} Vencer: {prob_vitoria_B*100:.2f}%  ->  (Odd Justa Estimada: {1/prob_vitoria_B:.2f})")
    print(f"Probabilidade Empate:       {prob_empate*100:.2f}%  ->  (Odd Justa Estimada: {1/prob_empate:.2f})")
    print("-" * 40)
    print("Conclusão para Apostas (EV+):")
    print("Compare as 'Odds Justas Estimadas' acima com as oferecidas na casa de aposta.")
    print("Se a Odd da casa for MAIOR que a sua Odd Justa, ali existe Valor Esperado Positivo (EV+).")
else:
    print("Execute as células anteriores para gerar a matriz final.")
""")

with open('seminario_copa_2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook gerado com sucesso!")
