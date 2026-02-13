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
    "BBDC4.SA", "ABEV3.SA", "RADL3.SA", "BBAS3.SA", "ALPA4.SA", 
        "VIVA3.SA", "ENEV3.SA", "USIM5.SA", "SUZB3.SA", "VIVT3.SA", 
        "RANI3.SA", "JHSF3.SA", "GMAT3.SA", "WIZC3.SA", "CURY3.SA", 
        "INTB3.SA", "FLRY3.SA", "EZTC3.SA", "EGIE3.SA", 
        "SOL-USD", "TEND3.SA", "SLCE3.SA", "ITUB4.SA", "WEGE3.SA",
        "ENGI11.SA", "MULT3.SA", "DIRR3.SA", "BPAC11.SA", "CVCB3.SA", 
        "MOVI3.SA", "TIMS3.SA", "AMBP3.SA", "LWSA3.SA", "MGLU3.SA", 
        "IGTI11.SA", "HAPV3.SA", "RDOR3.SA", "CMIG4.SA", "EQTL3.SA", 
        "BBSE3.SA", "SANB11.SA", "POMO4.SA", "CSAN3.SA", "CYRE3.SA", 
        "YDUQ3.SA", "ETH-USD", "BRBI11.SA", "CPFE3.SA", "ASAI3.SA", 
        "MRVE3.SA", "UNIP6.SA", "VALE3.SA"
]
# --- LISTA TOP 20 SNIPER ---
TOP_20_SNIPER = [
    "RADL3.SA", "WIZC3.SA", "BBAS3.SA", "VIVA3.SA", "DIRR3.SA",
    "SOL-USD", "MGLU3.SA", "AMBP3.SA", "COGN3.SA", "MOVI3.SA",
    "EGIE3.SA", "BBSE3.SA", "TAEE11.SA", "SBSP3.SA", "CURY3.SA",
    "BRBI11.SA", "KEPL3.SA", "CMIG4.SA", "EZTC3.SA", "LREN3.SA"
]

# 2. AGORA O SEU CÓDIGO DA SIDEBAR VAI FUNCIONAR:
with st.sidebar:
    st.header("🎯 Seleção de Ativos")
    modo_selecao = st.radio("Modo de Scan:", ["Top 20 Sniper Lab", "Lista Base (52 Ativos)", "Manual"])
    
    if modo_selecao == "Top 20 Sniper Lab":
        tickers_para_scan = TOP_20_SNIPER
    elif modo_selecao == "Lista Base (52 Ativos)":
        tickers_para_scan = tickers_base
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

            last_row = df.iloc[-1]
            results.append({
                "Ticker": t.replace(".SA", ""),
                "Ticker_Full": t,
                "Preço": float(last_row['Close']),
                "IFR2": float(last_row['IFR2']),
                "ATR": float(last_row['ATR']),
                "MM200": "✅ ACIMA" if last_row['Close'] > last_row['SMA200'] else "❌ ABAIXO",
                "SINAL": "🔥 COMPRA" if (last_row['Close'] > last_row['SMA200'] and last_row['IFR2'] < ifr_inferior) else "AGUARDAR",
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

        cols_show = [c for c in df_ex.columns if c != "Ticker_Full"]
        st.dataframe(df_ex[cols_show].style.format({"Preço": "R$ {:.2f}", "Alvo": "R$ {:.2f}", "IFR2": "{:.2f}", "Potencial %": "{:.2f}%", "Vol Médio (M)": "{:.2f}M"}).map(lambda v: f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold' if v == "🔥 COMPRA" else '', subset=['SINAL']), use_container_width="stretch", hide_index=True)
        
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
    
    if 'Vol_vs_Media' in df_gringo.columns:
        # 1. CÁLCULOS GLOBAIS DE LIQUIDEZ
        vol_financeiro_hoje = (df_gringo['Preço'] * (df_gringo['Vol_Hoje (M)'] * 1_000_000)).sum() / 1_000_000
        vol_financeiro_medio = (df_gringo['Preço'] * (df_gringo['Vol Médio (M)'] * 1_000_000)).sum() / 1_000_000
        delta_vol_total = ((vol_financeiro_hoje / vol_financeiro_medio) - 1) * 100

        st.markdown("### 📊 Liquidez Global do Scan")
        cx1, cx2, cx3 = st.columns(3)
        
        cx1.metric("Volume Total Hoje", f"R$ {vol_financeiro_hoje:,.0f}M", 
                  delta=f"{delta_vol_total:.2f}%", help="Soma do volume financeiro hoje.")
        cx2.metric("Volume Total Médio", f"R$ {vol_financeiro_medio:,.0f}M")
        status_mercado = "🔥 Mercado quente" if delta_vol_total > 0 else "❄️ Mercado frio"
        cx3.metric("Status de Liquidez", status_mercado)

        st.divider()

        # 2. CARDS DE SENTIMENTO
        c1, c2, c3, c4 = st.columns(4)
        fluxo_medio = df_gringo['Vol_vs_Media'].mean()
        sentimento_obv = df_gringo['Fluxo_OBV'].value_counts().idxmax()
        
        c1.metric("Pressão de Volume", "Alta" if fluxo_medio > 1 else "Baixa", f"{((fluxo_medio-1)*100):.2f}%")
        c2.metric("Fluxo Majoritário", sentimento_obv)
        c3.metric("Ativos Escaneados", f"{len(df_gringo)}")
        
        # Correção aqui: Comprador em maiúsculo
        veredito = "✅ SEGUIR FLUXO" if (fluxo_medio > 1 and sentimento_obv == "Comprador") else "⚠️ CAUTELA"
        c4.markdown(f"**Veredito:** {veredito}")

        # 3. TABELA TOP 10 MÃO FORTE
        st.write("**Top 10 Ativos com maior volume em relação à média (Mão Forte):**")
        
        top_gringo = df_gringo[[
            'Ticker', 'Vol_Hoje (M)', 'Vol Médio (M)', 'Vol_vs_Media', 'Fluxo_OBV'
        ]].sort_values('Vol_vs_Media', ascending=False).head(10)
        
        # Correção da cor: x == "Comprador"
        st.table(top_gringo.style.format({
            "Vol_Hoje (M)": "{:.2f}M",
            "Vol Médio (M)": "{:.2f}M",
            "Vol_vs_Media": "{:.2f}x"
        }).applymap(lambda x: 'color: #39FF14; font-weight: bold' if x == "Comprador" else 'color: #D90429', subset=['Fluxo_OBV']))
# --- BACKTEST ---
with tab_back:
    st.subheader("🧪 Simulador de Estratégia (IFR2 + Filtros)")
    if st.session_state.dados_brutos is not None and st.session_state.df_resultado is not None:
        col_b1, col_b2 = st.columns([1, 2])
        mapa_bt = dict(zip(st.session_state.df_resultado['Ticker'], st.session_state.df_resultado['Ticker_Full']))
        
        with col_b1:
            ativo_bt = st.selectbox("Escolha o Ativo:", st.session_state.df_resultado['Ticker'].tolist(), key="bt_ativo")
            ifr_gatilho = st.number_input("Entrar se IFR2 <:", value=25)
            periodo_bt = st.selectbox("Simular nos últimos:", ["Todo o período (2 anos)", "12 meses", "6 meses", "3 meses"], index=0)
            filtro_mm = st.radio("Filtro de Tendência:", ["MM200 (Longo Prazo)", "MM52 (Trimestral)", "Sem Filtro (Agressivo)"], index=0)
            
            st.markdown("---")
            st.write("🛡️ **Gerenciamento de Risco**")
            ativar_stop_fixo = st.checkbox("Utilizar Stop Loss Fixo", value=True)
            perc_stop_bt = st.number_input("Distância do Stop (%)", value=5.0, disabled=not ativar_stop_fixo)
            usar_time_stop = st.checkbox("Usar Time Stop (5 dias)", value=True)

        t_bt = mapa_bt[ativo_bt]
        df_bt = st.session_state.dados_brutos[t_bt].copy() if len(mapa_bt) > 1 else st.session_state.dados_brutos.copy()
        
        df_bt['MM200'] = ta.sma(df_bt['Close'], 200)
        df_bt['SMA52'] = ta.sma(df_bt['Close'], 52)
        df_bt['IFR2'] = ta.rsi(df_bt['Close'], 2)
        df_bt['Alvo'] = df_bt['High'].shift(1).rolling(2).max()
        
        ultima_data = df_bt.index.max()
        if periodo_bt == "12 meses": df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=12))]
        elif periodo_bt == "6 meses": df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=6))]
        elif periodo_bt == "3 meses": df_bt = df_bt[df_bt.index >= (ultima_data - pd.DateOffset(months=3))]
        
        df_bt = df_bt.dropna(subset=['IFR2', 'Alvo'])

        trades_bt = []
        em_operacao = False
        
        for i in range(len(df_bt)):
            if not em_operacao:
                condicao_tendencia = True
                if filtro_mm == "MM200 (Longo Prazo)":
                    condicao_tendencia = df_bt['Close'].iloc[i] > df_bt['MM200'].iloc[i] if not pd.isna(df_bt['MM200'].iloc[i]) else False
                elif filtro_mm == "MM52 (Trimestral)":
                    condicao_tendencia = df_bt['Close'].iloc[i] > df_bt['SMA52'].iloc[i] if not pd.isna(df_bt['SMA52'].iloc[i]) else False
                
                if condicao_tendencia and df_bt['IFR2'].iloc[i] < ifr_gatilho:
                    p_entrada, d_entrada, em_operacao, dias_op = df_bt['Close'].iloc[i], df_bt.index[i], True, 0
            else:
                dias_op += 1
                p_high, p_low, p_close, v_alvo = df_bt['High'].iloc[i], df_bt['Low'].iloc[i], df_bt['Close'].iloc[i], df_bt['Alvo'].iloc[i]
                v_stop = p_entrada * (1 - perc_stop_bt/100)
                
                if p_high >= v_alvo:
                    res = (v_alvo / p_entrada) - 1
                    trades_bt.append({'Entrada': d_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'ALVO'})
                    em_operacao = False
                elif ativar_stop_fixo and p_low <= v_stop:
                    res = (v_stop / p_entrada) - 1
                    trades_bt.append({'Entrada': d_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'STOP'})
                    em_operacao = False
                elif usar_time_stop and dias_op >= 5:
                    res = (p_close / p_entrada) - 1
                    trades_bt.append({'Entrada': d_entrada, 'Saída': df_bt.index[i], 'Resultado %': res * 100, 'Status': 'TIME STOP'})
                    em_operacao = False

        if trades_bt:
            tdf = pd.DataFrame(trades_bt)
            tdf['Acumulado %'] = tdf['Resultado %'].cumsum()
            total_ret = tdf['Resultado %'].sum()
            win_rate = (tdf['Resultado %'] > 0).mean() * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno Acumulado", f"{total_ret:.2f}%")
            m2.metric("Taxa de Acerto", f"{win_rate:.1f}%")
            m3.metric("Total Trades", len(tdf))
            m4.metric("Avg. Trade", f"{(total_ret/len(tdf)):.2f}%")
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=tdf['Saída'], y=tdf['Acumulado %'], fill='tozeroy', line=dict(color=CORES_SNIPER['verde_neon'])))
            fig_bt.update_layout(title=f"Curva de Patrimônio: {ativo_bt}", template="plotly_dark", height=400)
            st.plotly_chart(fig_bt, use_container_width="stretch")
            
            with st.expander("Ver lista de operações"):
                # Formatação da tabela para incluir o %
                st.dataframe(tdf.style.format({
                    "Resultado %": "{:.2f}%",
                    "Acumulado %": "{:.2f}%"
                }).map(lambda x: f"color: {'#39FF14' if x > 0 else '#D90429'}", subset=['Resultado %', 'Acumulado %']), use_container_width="stretch")
        else: st.warning("Nenhum trade encontrado.")
    else: st.info("⚠️ Execute o SCAN primeiro.")

