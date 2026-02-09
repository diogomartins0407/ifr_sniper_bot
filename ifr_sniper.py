import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
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
    'roxo_mm52': '#BF40BF'
}

st.set_page_config(page_title="Sniper IFR2 - Terminal Quant", layout="wide")

# 2. PROCESSAMENTO DE DADOS
@st.cache_data(ttl=3600)
def processar_dados_sniper(tickers):
    data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
    results = []
    
    for t in tickers:
        try:
            df = data[t].copy().dropna()
            if len(df) < 200: continue 

            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['SMA52'] = ta.sma(df['Close'], length=52) # ADICIONADA MM52
            df['SMA20'] = ta.sma(df['Close'], length=20)
            df['IFR2'] = ta.rsi(df['Close'], length=2)
            df['Alvo'] = df['High'].shift(1).rolling(window=2).max()
            df['Vol_Medio'] = df['Volume'].rolling(window=21).mean()

            last_row = df.iloc[-1]
            preco_at = float(last_row['Close'])
            mm200_v = float(last_row['SMA200'])
            mm52_v = float(last_row['SMA52'])
            ifr2_v = float(last_row['IFR2'])
            alvo_v = float(last_row['Alvo'])
            
            sinal = "🔥 COMPRA" if (preco_at > mm200_v and ifr2_v < 25) else "AGUARDAR"
            potencial = ((alvo_v / preco_at) - 1) * 100
            
            results.append({
                "Ticker": t.replace(".SA", ""),
                "Preço": round(preco_at, 2),
                "IFR2": round(ifr2_v, 2),
                "MM200": "✅ ACIMA" if preco_at > mm200_v else "❌ ABAIXO",
                "MM52": round(mm52_v, 2), # COLUNA MM52
                "SINAL": sinal,
                "Alvo": round(alvo_v, 2),
                "Potencial %": round(potencial, 2),
                "Vol Médio (M)": round(last_row['Vol_Medio'] / 1_000_000, 2),
                "Data": last_row.name.strftime('%d/%m/%Y'),
            })
        except: continue
    return pd.DataFrame(results), data

# 3. UI PRINCIPAL
st.title("🎯 Sniper IFR2 | Terminal Quant")

tickers_sniper = [
    "PETR4.SA", "VALE3.SA", "CYRE3.SA", "BBDC4.SA", "LREN3.SA", 
    "AZZA3.SA", "EQTL3.SA", "DIRR3.SA", "CURY3.SA", "STBP3.SA", 
    "ITUB4.SA", "BBAS3.SA", "CEAB3.SA", "PRIO3.SA", "ELET3.SA", 
    "WEGE3.SA", "BPAC11.SA", "CMIG4.SA", "CSNA3.SA", "JBSS3.SA"
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
        compras = df_ex[df_ex['SINAL'] == "🔥 COMPRA"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Ativos", len(df_ex))
        m2.metric("Oportunidades", len(compras))
        m3.metric("Último Fechamento", df_ex['Data'].iloc[0])

        st.write("### 📋 Grade de Cotações")
        st.dataframe(
            df_ex.style.format({
                "Preço": "R$ {:.2f}", "Alvo": "R$ {:.2f}", "MM52": "R$ {:.2f}",
                "IFR2": "{:.2f}", "Potencial %": "{:.2f}%", "Vol Médio (M)": "{:.2f}M"
            }).applymap(lambda v: f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold' if v == "🔥 COMPRA" else '', subset=['SINAL']),
            use_container_width=True, hide_index=True
        )

        st.write("---")
        escolha = st.selectbox("Análise Gráfica Detalhada:", df_ex['Ticker'].tolist())
        if escolha:
            f_df = st.session_state.dados_brutos[escolha + ".SA"].copy()
            f_df['SMA200'] = ta.sma(f_df['Close'], length=200)
            f_df['SMA52'] = ta.sma(f_df['Close'], length=52) # ADICIONADA AO GRÁFICO
            f_df['SMA20'] = ta.sma(f_df['Close'], length=20)
            p_df = f_df.tail(150)
            
            fig = go.Figure(data=[go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='Preço')])
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA200'], line=dict(color=CORES_SNIPER['laranja_mm200'], width=2), name='Média 200 (Longo Prazo)'))
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA52'], line=dict(color=CORES_SNIPER['roxo_mm52'], width=2), name='Média 52 (Trimestral)'))
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color=CORES_SNIPER['azul_selecao'], dash='dot'), name='Média 20 (Curto Prazo)'))
            
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aguardando execução do Scan...")

with tab_calc:
    st.subheader("🧮 Calculadora de Posição")
    capital = st.number_input("Capital Total (R$)", value=10000.0, step=1000.0)
    risco_perc = st.slider("Risco por Operação (%)", 0.5, 5.0, 1.0)
    
    if st.session_state.df_resultado is not None:
        ativo_calc = st.selectbox("Selecione o Ativo para Cálculo:", df_ex['Ticker'].tolist())
        preco_calc = df_ex[df_ex['Ticker'] == ativo_calc]['Preço'].values[0]
        stop_sugerido = preco_calc * 0.95 # Exemplo: 5% de stop
        
        perda_financeira = capital * (risco_perc/100)
        num_acoes = perda_financeira / (preco_calc - stop_sugerido)
        
        c1, c2 = st.columns(2)
        c1.metric("Qtd. Ações", int(num_acoes))
        c2.metric("Financeiro Alocado", f"R$ {int(num_acoes) * preco_calc:.2f}")