import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 1. IDENTIDADE VISUAL "DEEP QUANT"
CORES_SNIPER = {
    'bg_deep': '#0D1117',
    'text': '#E6EDF3',
    'verde_neon': '#39FF14',
    'vermelho': '#D90429',
    'cinza_fundo': '#161B22',
    'azul_selecao': '#58A6FF',
    'laranja_mm200': '#FFA500',
    'roxo_mm52': '#BF40BF',
    'cinza_grid': '#30363D',
    'vermelho_transparente': 'rgba(217, 4, 41, 0.15)',
    'verde_transparente': 'rgba(57, 255, 20, 0.1)'
}

st.set_page_config(page_title="Sniper IFR2", layout="wide")

# --- CONTROLES LATERAIS (SIDEBAR) ---
# --- CONTROLES LATERAIS (SIDEBAR) ---
st.sidebar.header("⚙️ Configurações do Gráfico")

# Trocando Sliders por Number Inputs para precisão total
ifr_superior = st.sidebar.number_input("Limite Superior IFR", min_value=50, max_value=95, value=70, step=1)
ifr_inferior = st.sidebar.number_input("Limite Inferior IFR", min_value=5, max_value=50, value=30, step=1)

# --- ADICIONAR NOVO ATIVO ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Buscar Novo Ativo")
novo_ticker = st.sidebar.text_input("Ex: PETR4 ou ^BVSP (Para B3)", "").upper() + ".SA" # Adiciona .SA automaticamente
if st.sidebar.button("➕ Adicionar Ativo"):
    if novo_ticker and novo_ticker not in st.session_state.tickers_adicionados:
        st.session_state.tickers_adicionados.add(novo_ticker)
        st.sidebar.success(f"{novo_ticker.replace('.SA', '')} adicionado!")
        # Força o re-execução do scan ao adicionar um ativo
        st.session_state.df_resultado = None 
        st.session_state.dados_brutos = None
    else:
        st.sidebar.warning("Ativo já na lista ou inválido.")

# Inicializar o conjunto de tickers adicionados na primeira execução
if 'tickers_adicionados' not in st.session_state:
    st.session_state.tickers_adicionados = set()

# 2. PROCESSAMENTO DE DADOS
@st.cache_data(ttl=3600)
def processar_dados_sniper(tickers):
    data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
    results = []
    
    for t in tickers:
        try:
            df = data[t].copy().dropna()
            if len(df) < 200: continue 

            # Cálculos Técnicos
            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['SMA52'] = ta.sma(df['Close'], length=52)
            df['SMA20'] = ta.sma(df['Close'], length=20)
            df['IFR2'] = ta.rsi(df['Close'], length=2)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['Alvo'] = df['High'].shift(1).rolling(window=2).max()
            df['Vol_Medio'] = df['Volume'].rolling(window=21).mean()

            # GARANTIA: Removemos qualquer linha que tenha valor nulo nos indicadores
            df_limpo = df.dropna()
            if df_limpo.empty: continue
            
            last_row = df_limpo.iloc[-1]
            
            results.append({
                "Ticker": t.replace(".SA", ""),
                "Preço": float(last_row['Close']),
                "IFR2": float(last_row['IFR2']),
                "ATR": float(last_row['ATR']),
                "MM200": "✅ ACIMA" if float(last_row['Close']) > float(last_row['SMA200']) else "❌ ABAIXO",
                "SINAL": "🔥 COMPRA" if (float(last_row['Close']) > float(last_row['SMA200']) and float(last_row['IFR2']) < 25) else "AGUARDAR",
                "Alvo": float(last_row['Alvo']),
                "Potencial %": ((float(last_row['Alvo']) / float(last_row['Close'])) - 1) * 100,
                "Vol Médio (M)": float(last_row['Vol_Medio']) / 1_000_000,
                "Data": last_row.name.strftime('%d/%m/%Y'),
            })
        except: continue
    return pd.DataFrame(results), data

# 3. UI PRINCIPAL
st.title("🎯 Sniper IFR2")

tickers_sniper = [
    "PETR4.SA", "VALE3.SA", "CYRE3.SA", "BBDC4.SA", "LREN3.SA", 
    "AZZA3.SA", "EQTL3.SA", "DIRR3.SA", "CURY3.SA", "STBP3.SA", 
    "ITUB4.SA", "BBAS3.SA", "CEAB3.SA", "PRIO3.SA", "ELET3.SA", 
    "WEGE3.SA", "BPAC11.SA", "CMIG4.SA", "CSNA3.SA", "JBSS3.SA", 
    "B3SA3.SA"
]

if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None
    st.session_state.dados_brutos = None
    st.session_state.tickers_adicionados = set()

if st.sidebar.button('🚀 EXECUTAR SCAN'):
    with st.spinner('Escaneando mercado...'):
        todos_os_tickers = list(set(tickers_sniper).union(st.session_state.tickers_adicionados))
        df_f, d_brutos = processar_dados_sniper(todos_os_tickers) # Passa a lista combinada
        st.session_state.df_resultado = df_f
        st.session_state.dados_brutos = d_brutos

tab_mon, tab_calc, tab_back = st.tabs(["📊 Monitoramento", "🧮 Calculadora de Risco", "🧪 Backtest por Ativo"])

with tab_mon:
    if st.session_state.df_resultado is not None:
        df_ex = st.session_state.df_resultado
        st.dataframe(
            df_ex.style.format({
                "Preço": "R$ {:.2f}", "Alvo": "R$ {:.2f}", "ATR": "{:.2f}",
                "IFR2": "{:.2f}", "Potencial %": "{:.2f}%", "Vol Médio (M)": "{:.2f}M"
            }).map(lambda v: f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold' if v == "🔥 COMPRA" else '', subset=['SINAL']),
            use_container_width=True, hide_index=True
        )

        st.write("---")
        escolha = st.selectbox("Análise Gráfica:", df_ex['Ticker'].tolist())
        if escolha:
            f_df = st.session_state.dados_brutos[escolha + ".SA"].copy()
            f_df['SMA200'] = ta.sma(f_df['Close'], length=200)
            f_df['SMA52'] = ta.sma(f_df['Close'], length=52) 
            f_df['SMA20'] = ta.sma(f_df['Close'], length=20)
            f_df['IFR2'] = ta.rsi(f_df['Close'], length=2)
            p_df = f_df.tail(100)
            
            # Criando Subplots: Preço (row 1, 50%), IFR2 (row 2, 25%), Volume (row 3, 25%)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, # Espaçamento menor entre os gráficos
                               row_heights=[0.6, 0.2, 0.2]) # Proporções de altura

            # --- GRÁFICO 1: PREÇO ---
            fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA200'], line=dict(color=CORES_SNIPER['laranja_mm200'], width=2), name='MM 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA52'], line=dict(color=CORES_SNIPER['roxo_mm52'], width=2), name='MM 52'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color=CORES_SNIPER['azul_selecao'], dash='dot'), name='MM 20'), row=1, col=1)

             # --- GRÁFICO 2: VOLUME ---
            # Cores do volume: verde se fechamento > abertura, vermelho se fechamento < abertura
            # Garante que as cores do volume correspondem exatamente aos dados do DataFrame
            volume_colors = ['rgba(57, 255, 20, 0.1)' if c >= o else 'rgba(217, 4, 41, 0.15)' for c, o in zip(p_df['Close'], p_df['Open'])]
            fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], name='Volume', marker=dict(color=volume_colors)), row=2, col=1)
            
            # --- GRÁFICO 3: IFR2 ---
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['IFR2'], line=dict(color=CORES_SNIPER['text'], width=1.5), name='IFR2'), row=3, col=1)
            fig.add_hline(y=ifr_superior, line_dash="dash", line_color=CORES_SNIPER['vermelho'], row=3, col=1)
            fig.add_hline(y=ifr_inferior, line_dash="dash", line_color=CORES_SNIPER['verde_neon'], row=3, col=1)
            fig.add_hrect(y0=ifr_superior, y1=100, fillcolor=CORES_SNIPER['vermelho_transparente'], line_width=0, row=3, col=1)
            fig.add_hrect(y0=0, y1=ifr_inferior, fillcolor=CORES_SNIPER['verde_transparente'], line_width=0, row=3, col=1)

           
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=800, hovermode='x unified')
            fig.update_yaxes(title_text="Volume", showgrid=False, row=2, col=1, autorange="reversed") # Eixo Y invertido para volume
            fig.update_yaxes(title_text="IFR", showgrid=False, row=3, col=1)
            fig.update_yaxes(title_text="Preço", showgrid=False, row=1, col=1)

            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Execute o SCAN.")

with tab_calc:
    st.subheader("🧮 Calculadora Sniper")
    if st.session_state.df_resultado is not None:
        df_calc = st.session_state.df_resultado
        c1, c2 = st.columns(2)
        
        with c1:
            capital = st.number_input("Capital Total (R$)", value=10000.0, step=1000.0)
            risco_perc = st.slider("Risco Total (%)", 0.1, 5.0, 1.0)
            ativo_calc = st.selectbox("Selecione o Ativo:", df_calc['Ticker'].tolist())
            
        # BUSCA SEGURA DOS DADOS (Evita KeyError)
        dados_ativos = df_calc[df_calc['Ticker'] == ativo_calc].iloc[0].to_dict()
        p_entrada = float(dados_ativos.get('Preço', 0))
        v_atr = float(dados_ativos.get('ATR', 0))
        
        with c2:
            mult_atr = st.number_input("Multiplicador ATR", value=2.0, step=0.5)
            financeiro_em_risco = capital * (risco_perc / 100)
            dist_stop = v_atr * mult_atr
            p_stop = p_entrada - dist_stop
            risco_unitario = p_entrada - p_stop
            
            if risco_unitario > 0:
                qtd_acoes = int(financeiro_em_risco / risco_unitario)
                financeiro_alocado = qtd_acoes * p_entrada
            else:
                qtd_acoes = 0
                financeiro_alocado = 0.0

        st.markdown("---")
        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Stop Loss", f"R$ {p_stop:.2f}")
        res2.metric("Quantidade", f"{qtd_acoes} un")
        res3.metric("Risco Financeiro", f"R$ {financeiro_em_risco:.2f}")
        res4.metric("Alocação Total", f"R$ {financeiro_alocado:.2f}")
    else:
        st.warning("⚠️ Execute o Scan primeiro.")

with tab_back:
    st.subheader("🧪 Simulador de Estratégia (IFR2 + Filtros de Tendência)")
    
    if st.session_state.dados_brutos is not None and st.session_state.df_resultado is not None:
        col_b1, col_b2 = st.columns([1, 2])
        
        with col_b1:
            tickers_disponiveis = st.session_state.df_resultado['Ticker'].tolist()
            ativo_bt = st.selectbox("Escolha o Ativo para Testar:", tickers_disponiveis, key="bt_ativo")
            
            ifr_gatilho = st.number_input("Entrar quando IFR2 for abaixo de:", value=25, step=1)
            
            st.markdown("---")
            st.write("📅 **Janela de Teste**")
            periodo_bt = st.selectbox("Simular nos últimos:", 
                                     ["Todo o período (2 anos)", "12 meses", "6 meses", "3 meses"],
                                     index=0)

            st.markdown("---")
            st.write("📈 **Filtro de Tendência**")
            filtro_mm = st.radio("Operar somente quando acima de:", 
                                 ["MM200 (Longo Prazo)", "MM52 (Trimestral)", "Sem Filtro (Agressivo)"],
                                 index=0, key="filtro_mm_radio")
            
            st.markdown("---")
            st.write("🛡️ **Gerenciamento de Risco**")
            usar_stop_fixo = st.checkbox("Usar Stop Fixo (%)", value=True, key="stop_fixo_check")
            perc_stop = st.number_input("Distância do Stop (%)", value=5.0, step=0.5) if usar_stop_fixo else 100.0
            usar_time_stop = st.checkbox("Usar Time Stop (5 dias)", value=True, key="time_stop_check")

        # --- EXTRAÇÃO SEGURA DOS DADOS (SEM AMBIGUIDADE) ---
        ticker_completo = ativo_bt + ".SA"
        
        # Tentativa 1: Ticker com .SA
        dados_raw = st.session_state.dados_brutos.get(ticker_completo)
        # Tentativa 2: Ticker puro (caso esteja assim no dict)
        if dados_raw is None:
            dados_raw = st.session_state.dados_brutos.get(ativo_bt)

        if dados_raw is not None:
            # Criamos DataFrame de trabalho limpo
            df_bt = pd.DataFrame(index=dados_raw.index)
            for c in ['Open', 'High', 'Low', 'Close']:
                if c in dados_raw.columns: 
                    df_bt[c] = dados_raw[c].values
                elif (ticker_completo, c) in dados_raw.columns: 
                    df_bt[c] = dados_raw[(ticker_completo, c)].values
            
            df_bt.columns = [col.capitalize() for col in df_bt.columns]
            df_bt = df_bt.dropna(subset=['Close'])

            # 1. CALCULAMOS INDICADORES NO HISTÓRICO COMPLETO (Para MM200 não ser NaN)
            df_bt['MM200'] = ta.sma(df_bt['Close'], length=200)
            df_bt['SMA52'] = ta.sma(df_bt['Close'], length=52)
            df_bt['IFR2'] = ta.rsi(df_bt['Close'], length=2)
            df_bt['Alvo'] = df_bt['High'].shift(1).rolling(window=2).max()
            
            # 2. FILTRAMOS O PERÍODO PARA A SIMULAÇÃO
            ultima_data = df_bt.index.max()
            if periodo_bt == "12 meses":
                df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=12))]
            elif periodo_bt == "6 meses":
                df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=6))]
            elif periodo_bt == "3 meses":
                df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=3))]
            
            df_bt = df_bt.dropna(subset=['IFR2', 'Alvo'])

            if not df_bt.empty:
                trades = []
                em_operacao = False
                
                for i in range(len(df_bt)):
                    if not em_operacao:
                        condicao_tendencia = True
                        val_mm200 = df_bt['MM200'].iloc[i]
                        val_mm52 = df_bt['SMA52'].iloc[i]

                        if filtro_mm == "MM200 (Longo Prazo)":
                            condicao_tendencia = float(df_bt['Close'].iloc[i]) > float(val_mm200) if not pd.isna(val_mm200) else False
                        elif filtro_mm == "MM52 (Trimestral)":
                            condicao_tendencia = float(df_bt['Close'].iloc[i]) > float(val_mm52) if not pd.isna(val_mm52) else False
                        
                        if condicao_tendencia and float(df_bt['IFR2'].iloc[i]) < ifr_gatilho:
                            preco_entrada = float(df_bt['Close'].iloc[i])
                            data_entrada = df_bt.index[i]
                            em_operacao = True
                            dias_na_operacao = 0
                    else:
                        dias_na_operacao += 1
                        p_high = float(df_bt['High'].iloc[i])
                        p_low = float(df_bt['Low'].iloc[i])
                        p_close = float(df_bt['Close'].iloc[i])
                        v_alvo = float(df_bt['Alvo'].iloc[i])
                        v_stop = preco_entrada * (1 - perc_stop/100)
                        
                        if p_high >= v_alvo:
                            res = (v_alvo / preco_entrada) - 1
                            trades.append({'Entrada': data_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'ALVO'})
                            em_operacao = False
                        elif usar_stop_fixo and p_low <= v_stop:
                            res = (v_stop / preco_entrada) - 1
                            trades.append({'Entrada': data_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'STOP'})
                            em_operacao = False
                        elif usar_time_stop and dias_na_operacao >= 5:
                            res = (p_close / preco_entrada) - 1
                            trades.append({'Entrada': data_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'TIME STOP'})
                            em_operacao = False

                if trades:
                    df_trades = pd.DataFrame(trades)
                    df_trades['Acumulado'] = df_trades['Resultado %'].cumsum()
                    
                    total_ret = df_trades['Resultado %'].sum()
                    win_rate = (df_trades['Resultado %'] > 0).mean() * 100
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Retorno Acumulado", f"{total_ret:.2f}%")
                    m2.metric("Taxa de Acerto", f"{win_rate:.1f}%")
                    m3.metric("Total Trades", len(df_trades))
                    m4.metric("Avg. Trade", f"{(total_ret/len(df_trades)):.2f}%")
                    
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=df_trades['Saída'], y=df_trades['Acumulado'], fill='tozeroy', line=dict(color=CORES_SNIPER['verde_neon'])))
                    fig_bt.update_layout(title=f"Curva de Patrimônio ({periodo_bt})", template="plotly_dark", height=400)
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    with st.expander("Ver lista de operações"):
                        def color_positive_negative(val):
                            color = '#39FF14' if val > 0 else '#D90429'
                            return f'color: {color}'
                        st.dataframe(df_trades.style.applymap(color_positive_negative, subset=['Resultado %', 'Acumulado']).format({
                            'Resultado %': '{:.2f}%',
                            'Acumulado': '{:.2f}%'
                        }))
                else:
                    st.warning(f"Nenhum trade encontrado nos parâmetros e período selecionado.")
            else:
                st.error("Dados insuficientes para o período selecionado.")
    else:
        st.info("⚠️ Execute o SCAN na aba Monitoramento para habilitar o Backtest.")
