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

# --- ESTADO DA SESSÃO ---
if 'tickers_adicionados' not in st.session_state:
    st.session_state.tickers_adicionados = set()
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None
if 'dados_brutos' not in st.session_state:
    st.session_state.dados_brutos = None

# LISTA DE ATIVOS
tickers_base = [
    "ASAI3.SA", "ALPA4.SA", "FLRY3.SA", "ENEV3.SA", "BRBI11.SA", 
    "CVCB3.SA", "EZTC3.SA", "VALE3.SA", "HASH11.SA", "VIVT3.SA", 
    "BBDC4.SA", "JHSF3.SA", "WIZC3.SA", "USIM5.SA", "VIVA3.SA", 
    "RADL3.SA", "AZZA3.SA", "ABEV3.SA", "TEND3.SA", "MGLU3.SA", 
    "SUZB3.SA", "CURY3.SA", "LWSA3.SA", "AMBP3.SA", "BRKM5.SA", 
    "RANI3.SA", "MRVE3.SA", "VAMO3.SA", "GMAT3.SA", "IGTI11.SA", 
    "KEPL3.SA", "EGIE3.SA", "DIRR3.SA", "MULT3.SA", "MOVI3.SA", 
    "RAIZ4.SA", "AXIA3.SA", "BPAC11.SA", "VBBR3.SA", "EQTL3.SA", 
    "SLCE3.SA", "RDOR3.SA", "CPLE3.SA", "CEAB3.SA", "BBAS3.SA", 
    "LREN3.SA", "MOTV3.SA", "PETR4.SA", "PSSA3.SA", "RECV3.SA",
    "WEGE3.SA"
]
# --- LISTA TOP 20 SNIPER ---
TOP_20_SNIPER = [
    "ASAI3.SA", "ALPA4.SA", "FLRY3.SA", "ENEV3.SA", "BRBI11.SA", 
    "CVCB3.SA", "EZTC3.SA", "PETR4.SA", "HASH11.SA", "VIVT3.SA", 
    "BBDC4.SA", "JHSF3.SA", "WIZC3.SA", "USIM5.SA", "VIVA3.SA", 
    "RADL3.SA", "AZZA3.SA", "ABEV3.SA", "TEND3.SA", "VALE3.SA"
]

LISTA_OURO_STORMER = [
    "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "ITUB4.SA", "PETR4.SA", 
    "VALE3.SA", "RADL3.SA", "RENT3.SA", "VIVT3.SA", "ELET3.SA",
    "WEGE3.SA", "GGBR4.SA", "PRIO3.SA", "EQTL3.SA", "SBSP3.SA",
    "LREN3.SA", "CCRO3.SA", "JBSS3.SA", "B3SA3.SA", "UGPA3.SA"
]

# 2. CÓDIGO DA SIDEBAR
with st.sidebar:
    st.header("🎯 Seleção de Ativos")
    modo_selecao = st.radio("Modo de Scan:", ["Top 20 Sniper Lab", "Lista Base", "Lista Ouro Stormer", "Manual"])
    
    if modo_selecao == "Top 20 Sniper Lab":
        tickers_para_scan = TOP_20_SNIPER
    elif modo_selecao == "Lista Base":
        tickers_para_scan = tickers_base
    elif modo_selecao == "Lista Ouro Stormer":
        tickers_para_scan = LISTA_OURO_STORMER
    else:
        raw_input = st.text_area("Insira os tickers (um por linha):")
        tickers_para_scan = [t.strip().upper() for t in raw_input.split('\n') if t.strip()]

    # Botão de Scan dentro da Sidebar para ficar organizado
    botao_scan = st.button('🚀 EXECUTAR SCAN', key='btn_principal_scan')

# --- CONTROLES LATERAIS (SIDEBAR) ---
st.sidebar.header("⚙️ Configurações do Gráfico")
ifr_superior = st.sidebar.number_input("Limite Superior IFR", min_value=50, max_value=95, value=70, step=1)
ifr_inferior = st.sidebar.number_input("Limite Inferior IFR", min_value=5, max_value=50, value=25, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Buscar Novo Ativo")
input_ticker = st.sidebar.text_input("Ex: PETR4", "").upper().strip()

if st.sidebar.button("➕ Adicionar Ativo"):
    if input_ticker:
        final_t = input_ticker if ("-" in input_ticker or "USD" in input_ticker) else f"{input_ticker}.SA"
        st.session_state.tickers_adicionados.add(final_t)
        st.sidebar.success(f"{final_t} adicionado!")
        st.session_state.df_resultado = None 

# --- FUNÇÃO DE MINI-BACKTEST (Alta Velocidade) ---
def fast_winrate(df, ifr_gatilho):
    """Calcula o WR agressivo (Sem Filtro, Sem MM5, Time Stop 5) p/ o IFR atual"""
    trades = []
    em_op = False
    p_entrada = 0
    dias_op = 0
    
    # Otimização com itertuples para não atrasar o Scan
    for row in df.itertuples():
        if not em_op:
            if row.IFR2 <= ifr_gatilho: # Só entra se o IFR bater no nível atual
                p_entrada = row.Close
                em_op = True
                dias_op = 0
        else:
            dias_op += 1
            if row.Open >= row.Alvo:
                trades.append((row.Open / p_entrada) - 1)
                em_op = False
            elif row.High >= row.Alvo:
                trades.append((row.Alvo / p_entrada) - 1)
                em_op = False
            elif dias_op >= 5:
                trades.append((row.Close / p_entrada) - 1)
                em_op = False
    
    if not trades: return 0.0, 0
    wins = sum(1 for t in trades if t > 0)
    return (wins / len(trades)) * 100, len(trades)

# 2. MOTOR DE PROCESSAMENTO
@st.cache_data(ttl=3600)
def processar_dados_sniper(tickers):
    data = yf.download(tickers, period="3y", interval="1d", group_by='ticker', progress=False)
    results = []
    
    for t in tickers:
        try:
            df = data[t].copy() if len(tickers) > 1 else data.copy()
            df = df.dropna(subset=['Close'])
            if len(df) < 200: continue 

            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['SMA52'] = ta.sma(df['Close'], length=52)
            df['SMA20'] = ta.sma(df['Close'], length=20)
            df['SMA5'] = ta.sma(df['Close'], length=5)
            df['IFR2'] = ta.rsi(df['Close'], length=2)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['Alvo'] = df['High'].shift(1).rolling(window=2).max()
            df['Vol_Medio'] = df['Volume'].rolling(window=21).mean()
            df['Vol_20'] = df['Volume'].rolling(20).mean()
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['OBV_Media'] = df['OBV'].rolling(10).mean()
            # Fluxo gringo
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            # Z-Score de Volume (Anomalia Estatística)
            v_mean = df['Volume'].rolling(20).mean()
            v_std = df['Volume'].rolling(20).std()
            df['Z_Vol'] = (df['Volume'] - v_mean) / v_std

            df_clean = df.dropna(subset=['IFR2', 'Alvo'])
            last_row = df_clean.iloc[-1]
            
            # Executa o mini-backtest agressivo para o nível de IFR2 atual
            ifr_atual = float(last_row['IFR2'])
            wr_hist, trades_hist = fast_winrate(df_clean, ifr_atual)
            
            sinal_hoje = "🔥 COMPRA" if (last_row['Close'] > last_row['SMA200'] and ifr_atual < ifr_inferior) else "AGUARDAR"

            results.append({
                "Ticker": t.replace(".SA", ""),
                "Ticker_Full": t,
                "Preço": float(last_row['Close']),
                "IFR2": ifr_atual,
                "ATR": float(last_row['ATR']),
                "MM200": "✅ ACIMA" if last_row['Close'] > last_row['SMA200'] else "❌ ABAIXO",
                "SINAL": sinal_hoje,
                "WR no Nível (3y)": f"{wr_hist:.1f}% ({trades_hist}t)",
                "Alvo": float(last_row['Alvo']),
                "Potencial %": ((float(last_row['Alvo']) / float(last_row['Close'])) - 1) * 100,
                "Vol Médio (M)": float(last_row['Vol_Medio']) / 1_000_000,
                "Data": last_row.name.strftime('%d/%m/%Y'),
                "Vol_Hoje (M)": float(last_row['Volume']) / 1_000_000,
                "Vol_vs_Media": float(last_row['Volume'] / last_row['Vol_20']) if last_row['Vol_20'] > 0 else 1.0,
                "Fluxo_OBV": "Comprador" if last_row['OBV'] > last_row['OBV_Media'] else "Vendedor"
            })
        except: continue
    return pd.DataFrame(results), data



# --- TÍTULO DO DASHBOARD ---
st.title("🎯 Sniper IFR2")

if botao_scan:
    with st.spinner('Escaneando mercado...'):
        lista_final = list(set(tickers_para_scan).union(st.session_state.tickers_adicionados))
        
        df_f, d_brutos = processar_dados_sniper(lista_final)
        st.session_state.df_resultado = df_f
        st.session_state.dados_brutos = d_brutos

tab_mon, tab_back = st.tabs(["📊 Monitoramento", "🧪 Backtest por Ativo"])

# --- MONITORAMENTO ---
with tab_mon:
    if st.session_state.df_resultado is not None:
        df_ex = st.session_state.df_resultado
        if 'Ticker_Full' not in df_ex.columns:
            st.warning("⚠️ Execute o SCAN.")
            st.stop()

        # Define a ordem para a nova coluna ficar ao lado do SINAL
        cols_base = ['Ticker', 'Preço', 'IFR2', 'MM200', 'SINAL', 'WR no Nível (3y)', 'Alvo', 'Potencial %', 'ATR', 'Vol Médio (M)', 'Vol_Hoje (M)', 'Vol_vs_Media', 'Data', 'Fluxo_OBV']
        cols_show = [c for c in cols_base if c in df_ex.columns]
        
        # Função para pintar o WR de Verde se >= 70% ou Vermelho se < 50%
        def colorir_wr(val):
            try:
                perc = float(val.split('%')[0])
                if perc >= 70.0: return f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold'
                elif perc < 50.0: return f'color: {CORES_SNIPER["vermelho"]}'
            except: pass
            return ''

        st.dataframe(df_ex[cols_show].style.format({
            "Preço": "R$ {:.2f}", 
            "Alvo": "R$ {:.2f}", 
            "IFR2": "{:.2f}", 
            "Potencial %": "{:.2f}%", 
            "Vol Médio (M)": "{:.2f}M"
        }).map(lambda v: f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold' if v == "🔥 COMPRA" else '', subset=['SINAL'])
          .map(colorir_wr, subset=['WR no Nível (3y)']), 
          use_container_width="stretch", hide_index=True)

        st.write("---")
        mapa = dict(zip(df_ex['Ticker'], df_ex['Ticker_Full']))
        escolha = st.selectbox("Análise Gráfica:", df_ex['Ticker'].tolist())
        
        if escolha:
            t_real = mapa[escolha]
            f_df = st.session_state.dados_brutos[t_real].copy() if len(mapa) > 1 else st.session_state.dados_brutos.copy()
            f_df = f_df.dropna(subset=['Close'])
            f_df['SMA200'] = ta.sma(f_df['Close'], 200)
            f_df['SMA52'] = ta.sma(f_df['Close'], 52)
            f_df['SMA20'] = ta.sma(f_df['Close'], 20)
            f_df['SMA5'] = ta.sma(f_df['Close'], 5)
            f_df['IFR2'] = ta.rsi(f_df['Close'], length=2)
            p_df = f_df.tail(120)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # 1. PREÇO
            fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA200'], line=dict(color=CORES_SNIPER['laranja_mm200'], width=2), name='MM 200', connectgaps=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA52'], line=dict(color=CORES_SNIPER['roxo_mm52'], width=2), name='MM 52', connectgaps=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color=CORES_SNIPER['azul_selecao'], width=2, dash='dot'), name='MM 20', connectgaps=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA5'], line=dict(color=CORES_SNIPER['vermelho'], width=2, dash='dot'), name='MM 5', connectgaps=True), row=1, col=1)
            
            # 2. VOLUME REVERSO
            v_cols = [CORES_SNIPER['verde_neon'] if p_df['Close'].iloc[i] >= p_df['Close'].iloc[i-1] else CORES_SNIPER['vermelho'] for i in range(len(p_df))]            
            fig.add_trace(go.Bar(x=p_df.index, y=p_df['Volume'], marker=dict(color=v_cols, opacity=0.4), name='Volume'), row=2, col=1)
            fig.update_yaxes(autorange="reversed", row=2, col=1)

            # 3. IFR2
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['IFR2'], line=dict(color=CORES_SNIPER['text'], width=1.5), name='IFR2', connectgaps=True), row=3, col=1)
            fig.add_hline(y=ifr_superior, line_dash="dash", line_color=CORES_SNIPER['vermelho'], row=3, col=1)
            fig.add_hline(y=ifr_inferior, line_dash="dash", line_color=CORES_SNIPER['verde_neon'], row=3, col=1)
            fig.add_hrect(y0=ifr_superior, y1=100, fillcolor=CORES_SNIPER['vermelho_transparente'], line_width=0, row=3, col=1)
            fig.add_hrect(y0=0, y1=ifr_inferior, fillcolor=CORES_SNIPER['verde_transparente'], line_width=0, row=3, col=1)

            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=900, hovermode='x unified',
                hoverdistance=100, # Aumenta a sensibilidade para capturar o eixo
                hoverlabel=dict(
                    bgcolor="rgba(22, 27, 34, 0.9)",
                    font_size=13,
                    font_family="Monospace",
                    align="left",      # Alinha o texto à esquerda na caixa
                    namelength=-1
                )
            )
            fig.update_traces(
                hoverinfo="all",
                selector=dict(type='candlestick')
            )
            if "-USD" not in escolha: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, width="stretch")
    else: st.info("💡 Execute o SCAN para começar.")

    # --- RADAR DE FLUXO GRINGO ---
    st.divider()
    st.subheader("🌐 Fluxo Gringo")

    if st.session_state.df_resultado is not None:
        df_gringo = st.session_state.df_resultado
    
    # 1. CÁLCULO DE AGRESSIVIDADE (Baseado em MFI e Z-Score)
    # Ativos onde o volume é uma anomalia (Z-Score > 1.5)
        anomalias = df_gringo[df_gringo['Vol_vs_Media'] > 1.5]
    
        st.markdown("### 📊 Liquidez")
        cx1, cx2, cx3 = st.columns(3)
    
        vol_hoje = (df_gringo['Preço'] * (df_gringo['Vol_Hoje (M)'] * 1_000_000)).sum() / 1_000_000
        vol_medio = (df_gringo['Preço'] * (df_gringo['Vol Médio (M)'] * 1_000_000)).sum() / 1_000_000
        delta = ((vol_hoje / vol_medio) - 1) * 100

        cx1.metric("Volume Financeiro Hoje", f"R$ {vol_hoje:,.0f}M", f"{delta:.2f}%")
        cx2.metric("Anomalias Detectadas", f"{len(anomalias)} ativos", help="Ativos com volume > 1.5x a média")
        
        # Sentimento Global (Baseado na média do OBV dos ativos)
        sentimento_geral = df_gringo['Fluxo_OBV'].mode()[0]
        cx3.metric("Sentimento Majoritário", sentimento_geral)

        st.divider()

        # 2. TABELA DE RASTREAMENTO DE DINHEIRO (SMART MONEY)
        st.write("**Top 10 Ativos com maior rastro de Dinheiro Institucional:**")
        
        # Criamos um Score de Fluxo (Volume + Sentimento)
        top_fluxo = df_gringo[[
            'Ticker', 'Vol_vs_Media', 'Fluxo_OBV', 'IFR2'
        ]].sort_values('Vol_vs_Media', ascending=False).head(10)

        # Nova lógica de Veredito para a tabela
        def diagnostico_fluxo(row):
            if row['Vol_vs_Media'] > 1.5 and row['Fluxo_OBV'] == "Comprador":
                return "🔥 ACUMULAÇÃO"
            elif row['Vol_vs_Media'] > 1.5 and row['Fluxo_OBV'] == "Vendedor":
                return "DISTRIBUIÇÃO"
            return "⏳ NEUTRO"

        top_fluxo['Diagnóstico'] = top_fluxo.apply(diagnostico_fluxo, axis=1)

        st.table(top_fluxo.style.format({
            "Vol_vs_Media": "{:.2f}x",
            "IFR2": "{:.2f}"
        }).applymap(lambda x: 'color: #39FF14; font-weight: bold' if x == "Comprador" or "ACUMULAÇÃO" in str(x) else 
                            'color: #D90429; font-weight: bold' if x == "Vendedor" or "DISTRIBUIÇÃO" in str(x) else '', 
                    subset=['Fluxo_OBV', 'Diagnóstico']))

# --- BACKTEST ---
with tab_back:
    st.subheader("🧪 Simulador de Estratégia")
    
    if st.session_state.dados_brutos is not None and st.session_state.df_resultado is not None:
        col_b1, col_b2 = st.columns([1, 2])
        mapa_bt = dict(zip(st.session_state.df_resultado['Ticker'], st.session_state.df_resultado['Ticker_Full']))
        
        with col_b1:
            ativo_bt = st.selectbox("Escolha o Ativo:", st.session_state.df_resultado['Ticker'].tolist(), key="bt_ativo")
            
            st.markdown("---")
            st.write("🎯 **Gatilho e Tendência**")
            ifr_gatilho = st.number_input("Entrar se IFR2 <:", value=25)
            filtro_tendencia = st.selectbox("Filtro de Tendência:", ["SMA200", "SMA52", "Sem Filtro"], index=0)
            periodo_bt = st.selectbox("Simular nos últimos:", ["Todo o período (2 anos)", "12 meses", "6 meses", "3 meses"], index=0)
            
            st.markdown("---")
            st.write("🛡️ **Gerenciamento de Risco**")
            usar_stop_mm5 = st.checkbox("Sair na MM5 (Se estiver no Lucro)", value=True)
            ativar_stop_fixo = st.checkbox("Utilizar Stop Loss Fixo", value=False)
            perc_stop_bt = st.number_input("Distância do Stop (%)", value=5.0, disabled=not ativar_stop_fixo)
            
            usar_time_stop = st.checkbox("Usar Time Stop", value=True)
            time_stop_val = st.slider("Dias Máx (Time Stop)", min_value=3, max_value=15, value=5, disabled=not usar_time_stop)

        with col_b2:
            # 1. PREPARAÇÃO DOS DADOS
            t_bt = mapa_bt[ativo_bt]
            df_bt = st.session_state.dados_brutos[t_bt].copy() if len(mapa_bt) > 1 else st.session_state.dados_brutos.copy()
            
            # Indicadores de Tendência e Saída (Viés de Futuro Corrigido)
            df_bt['SMA200'] = ta.sma(df_bt['Close'], 200)
            df_bt['SMA52'] = ta.sma(df_bt['Close'], 52)
            df_bt['SMA5_Prev'] = ta.sma(df_bt['Close'], 5).shift(1)
            df_bt['IFR2'] = ta.rsi(df_bt['Close'], 2)
            df_bt['Alvo'] = df_bt['High'].shift(1).rolling(2).max()
            
            # 2. FILTRAGEM DO PERÍODO
            ultima_data = df_bt.index.max()
            if periodo_bt == "12 meses": data_corte = ultima_data - pd.DateOffset(months=12)
            elif periodo_bt == "6 meses": data_corte = ultima_data - pd.DateOffset(months=6)
            elif periodo_bt == "3 meses": data_corte = ultima_data - pd.DateOffset(months=3)
            else: data_corte = df_bt.index.min()

            df_sim = df_bt[df_bt.index >= data_corte].copy()
            df_sim = df_sim.dropna(subset=['IFR2', 'Alvo', 'SMA5_Prev'])

            trades_bt = []
            em_operacao = False
            
            # 3. LOOP DE SIMULAÇÃO (MOTOR IDÊNTICO AO DO LAB)
            for i in range(len(df_sim)):
                row = df_sim.iloc[i]
                
                if not em_operacao:
                    # Aplica o Filtro de Tendência escolhido pelo usuário
                    cond_tendencia = True
                    if filtro_tendencia != "Sem Filtro":
                        val_mm = row[filtro_tendencia]
                        cond_tendencia = (row['Close'] > val_mm) if not pd.isna(val_mm) else False
                    
                    if cond_tendencia and row['IFR2'] < ifr_gatilho:
                        p_entrada = row['Close']
                        d_entrada = df_sim.index[i]
                        em_operacao = True
                        dias_op = 0
                else:
                    dias_op += 1
                    p_high, p_low, p_close, p_open = row['High'], row['Low'], row['Close'], row['Open']
                    v_alvo = row['Alvo']
                    v_stop = p_entrada * (1 - perc_stop_bt/100) if ativar_stop_fixo else 0
                    v_mm5 = row['SMA5_Prev']
                    
                    # A. GAPS DE ABERTURA (Prioridade 1)
                    if p_open >= v_alvo:
                        res = (p_open / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE ALTA (ALVO)'})
                        em_operacao = False
                        continue
                    elif ativar_stop_fixo and p_open <= v_stop:
                        res = (p_open / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE BAIXA (STOP)'})
                        em_operacao = False
                        continue
                    elif usar_stop_mm5 and v_mm5 > p_entrada and p_open <= v_mm5:
                        res = (p_open / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE BAIXA (STOP MM5)'})
                        em_operacao = False
                        continue

                    # B. EXECUÇÃO NO PREGÃO (Ordem Pessimista: Stop verificado ANTES do Alvo)
                    if ativar_stop_fixo and p_low <= v_stop:
                        res = (v_stop / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'STOP FIXO'})
                        em_operacao = False
                        continue
                        
                    elif usar_stop_mm5 and v_mm5 > p_entrada and p_low <= v_mm5:
                        res = (v_mm5 / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'STOP MM5'})
                        em_operacao = False
                        continue
                        
                    elif p_high >= v_alvo:
                        res = (v_alvo / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'ALVO ATINGIDO'})
                        em_operacao = False
                        continue

                    # C. TIME STOP
                    if usar_time_stop and dias_op >= time_stop_val:
                        res = (p_close / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'TIME STOP'})
                        em_operacao = False
                        continue
                    
                    # D. ENCERRAMENTO FORÇADO (Fim do arquivo)
                    if i == len(df_sim) - 1:
                        res = (p_close / p_entrada) - 1
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'FIM DOS DADOS'})
                        em_operacao = False 

            # 4. EXIBIÇÃO DOS RESULTADOS (Matemática Corrigida + Visual Original + Drawdown/Payoff)
            if trades_bt:
                tdf = pd.DataFrame(trades_bt)
                
                # CORREÇÃO MATEMÁTICA: Juros Compostos (Retorno Geométrico)
                tdf['Fator_Retorno'] = 1 + (tdf['Resultado %'] / 100)
                tdf['Acumulado Multiplicador'] = tdf['Fator_Retorno'].cumprod()
                tdf['Acumulado %'] = (tdf['Acumulado Multiplicador'] - 1) * 100
                
                # --- NOVAS LINHAS: CÁLCULO DE DRAWDOWN E PAYOFF ---
                tdf['Pico Acumulado'] = tdf['Acumulado Multiplicador'].cummax()
                tdf['Drawdown'] = (tdf['Acumulado Multiplicador'] / tdf['Pico Acumulado']) - 1
                max_dd = tdf['Drawdown'].min() * 100
                
                ganhos = tdf[tdf['Resultado %'] > 0]['Resultado %'].mean()
                perdas = tdf[tdf['Resultado %'] < 0]['Resultado %'].mean()
                payoff = abs(ganhos / perdas) if not pd.isna(perdas) and perdas != 0 else float('inf')
                # ----------------------------------------------------

                total_ret = tdf['Acumulado %'].iloc[-1]
                win_rate = (tdf['Resultado %'] > 0).mean() * 100
                avg_trade = tdf['Resultado %'].mean()
                
                # --- NOVA LINHA: 
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Retorno Acumulado", f"{total_ret:.2f}%")
                m2.metric("Taxa de Acerto", f"{win_rate:.1f}%")
                m3.metric("Avg. Trade", f"{avg_trade:.2f}%")
                m4.metric("Payoff", f"{payoff:.2f}")
                m5.metric("Max Drawdown", f"{max_dd:.2f}%")
                m6.metric("Total Trades", len(tdf))
                
                # VISUAL ORIGINAL (Plotly Dark + Neon)
                fig_bt = go.Figure()
                cor_linha = '#39FF14' if total_ret >= 0 else '#D90429'
                fig_bt.add_trace(go.Scatter(x=tdf['Saída'], y=tdf['Acumulado %'], fill='tozeroy', line=dict(color=cor_linha)))
                fig_bt.update_layout(
                    title=f"Curva de Patrimônio: {ativo_bt}", 
                    template="plotly_dark", 
                    height=350,
                    yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray')
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # --- NOVA LINHA: ADICIONANDO 'Drawdown' NA TABELA EXPANDIDA ---
                with st.expander("Ver lista de operações detalhada"):
                    st.dataframe(tdf[['Entrada', 'Saída', 'Status', 'Resultado %', 'Acumulado %', 'Drawdown']].style.format({
                        "Resultado %": "{:.2f}%",
                        "Acumulado %": "{:.2f}%",
                        "Drawdown": "{:.2f}%",
                        "Entrada": lambda t: t.strftime("%d/%m/%Y"),
                        "Saída": lambda t: t.strftime("%d/%m/%Y")
                    }).map(lambda x: f"color: {'#39FF14' if x > 0 else '#D90429'}", subset=['Resultado %', 'Acumulado %']), use_container_width=True)
            else: 
                st.warning("Nenhum trade encontrado para os parâmetros selecionados.")
    else: st.info("⚠️ Execute o SCAN primeiro para carregar os dados brutos.")
