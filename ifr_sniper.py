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

st.sidebar.markdown("---")

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

if st.sidebar.button('🚀 EXECUTAR SCAN'):
    with st.spinner('Escaneando mercado...'):
        df_f, d_brutos = processar_dados_sniper(tickers_sniper)
        st.session_state.df_resultado = df_f
        st.session_state.dados_brutos = d_brutos

tab_mon, tab_calc = st.tabs(["📊 Monitoramento", "🧮 Calculadora de Risco"])

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
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA200'], line=dict(color=CORES_SNIPER['laranja_mm200'], width=2), name='MM 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA52'], line=dict(color=CORES_SNIPER['roxo_mm52'], width=2), name='MM 52'), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color=CORES_SNIPER['azul_selecao'], dash='dot'), name='MM 20'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['IFR2'], line=dict(color=CORES_SNIPER['text'], width=1.5), name='IFR2'), row=2, col=1)
            fig.add_hline(y=ifr_superior, line_dash="dash", line_color=CORES_SNIPER['vermelho'], row=2, col=1)
            fig.add_hline(y=ifr_inferior, line_dash="dash", line_color=CORES_SNIPER['verde_neon'], row=2, col=1)
            fig.add_hrect(y0=ifr_superior, y1=100, fillcolor=CORES_SNIPER['vermelho_transparente'], line_width=0, row=2, col=1)
            fig.add_hrect(y0=0, y1=ifr_inferior, fillcolor=CORES_SNIPER['verde_transparente'], line_width=0, row=2, col=1)

            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=800, hovermode='x unified')
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
