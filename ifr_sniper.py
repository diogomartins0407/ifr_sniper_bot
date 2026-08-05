import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from b3_flow import (fetch_fluxo_b3, calcular_metricas_fluxo,
                     gerar_alertas_institucionais, construir_matriz_calendario,
                     estatisticas_calendario)

# 1. IDENTIDADE VISUAL "DEEP QUANT."
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
    "RANI3.SA", "VIVT3.SA", "EQTL3.SA", "ENEV3.SA",
    "IGTI11.SA", "TAEE11.SA", "EGIE3.SA", "CPFE3.SA",
    "CURY3.SA", "TIMS3.SA", "SLCE3.SA", "CMIG4.SA",
    "UGPA3.SA", "POMO4.SA", "ITUB4.SA", "BBSE3.SA",
    "RECV3.SA", "MULT3.SA", "HYPE3.SA", "PRIO3.SA"
]

LISTA_OURO_STORMER = [
    "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "ITUB4.SA", "PETR4.SA", 
    "VALE3.SA", "RADL3.SA", "RENT3.SA", "VIVT3.SA", "AXIA3.SA",
    "WEGE3.SA", "GGBR4.SA", "PRIO3.SA", "EQTL3.SA", "SBSP3.SA",
    "LREN3.SA", "MOTV3.SA", "JBSS3.SA", "B3SA3.SA", "UGPA3.SA"
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

    # Auto-refresh durante o pregao (re-roda a pagina a cada 2 min)
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh (2min)", value=False,
        help="Re-executa o SCAN automaticamente a cada 2 minutos durante o pregao. "
             "Util para acompanhar IFR em tempo real (com delay ~15min do yfinance)."
    )
    if auto_refresh:
        # Streamlit nativo: rerun a cada N segundos
        try:
            import streamlit.runtime.scriptrunner as _sr
            # Usa fragment do Streamlit para auto-refresh
            st.markdown(
                '<meta http-equiv="refresh" content="120">',
                unsafe_allow_html=True
            )
            st.caption("⏱️ Auto-refresh ativo - pagina re-carrega a cada 2min")
        except Exception:
            pass

    # Botao para limpar cache (uso institucional - forca refresh apos updates)
    if st.button('🧹 Limpar Cache', key='btn_clear_cache',
                 help="Forca refresh dos dados. Use apos atualizacoes do codigo."):
        st.cache_data.clear()
        st.session_state.df_resultado = None
        st.session_state.dados_brutos = None
        st.session_state.pop("fluxo_b3_df", None)
        st.success("Cache limpo. Execute o SCAN novamente.")

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
def fast_winrate(df, ifr_gatilho, time_stop=7):
    """
    Mini-backtest agressivo para calcular WR do SETUP OPERACIONAL (IFR <= gatilho).
    Stormer puro: sem filtro de tendencia, sem stop MM5.
    Default time_stop=7 (canonico Stormer / QuantBrasil).
    Fecha posicoes abertas no fim do historico (mark-to-market) para evitar
    inflar WR artificialmente.
    """
    trades = []
    em_op = False
    p_entrada = 0
    dias_op = 0
    ultimo_close = None
    
    for row in df.itertuples():
        ultimo_close = row.Close
        if not em_op:
            if row.IFR2 <= ifr_gatilho:
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
            elif dias_op >= time_stop:
                trades.append((row.Close / p_entrada) - 1)
                em_op = False
    
    # Fecha posicao aberta no fim do historico (mark-to-market)
    # Evita viesar o WR para cima ao "esconder" trades perdedores em aberto
    if em_op and ultimo_close is not None and p_entrada > 0:
        trades.append((ultimo_close / p_entrada) - 1)
    
    if not trades: return 0.0, 0
    wins = sum(1 for t in trades if t > 0)
    return (wins / len(trades)) * 100, len(trades)

# 2. MOTOR DE PROCESSAMENTO
@st.cache_data(ttl=120)  # Cache curto: 2 minutos (yfinance tem delay ~15min para B3)
def processar_dados_sniper(tickers, ifr_gatilho_wr=25, _ts_bucket=None,
                            _versao="v4_2026_06_01"):
    """
    Baixa dados crus + calcula indicadores + WR do setup operacional.
    SINAL e calculado FORA da funcao (dinamico, responde ao slider).

    _ts_bucket: bucket de 2 minutos (int(time.time()/120)) - forca cache a
    invalidar a cada 2 min mesmo se ifr_gatilho_wr nao mudar. Garante que
    novos SCANs busquem cotacoes atualizadas durante o pregao.
    """
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
            
            # === DUAS ESTATISTICAS COMPLEMENTARES ===
            ifr_atual = float(last_row['IFR2'])

            # WR @ Nivel: usa IFR atual do ativo como gatilho
            # Captura o "premio de sobrevenda" - quanto mais extremo o IFR,
            # mais alto tende a ser o WR (mean reversion classico)
            wr_nivel, n_nivel = fast_winrate(df_clean, ifr_atual, time_stop=7)

            # WR Setup: usa o limite operacional (ifr_inferior) como gatilho
            # Baseline comparavel entre ativos - todos sob o mesmo regime
            wr_setup, n_setup = fast_winrate(df_clean, ifr_gatilho_wr, time_stop=7)

            # Delta: premio de sobrevenda atual (positivo = momento estatistico mais favoravel)
            delta_wr = wr_nivel - wr_setup
            
            # SINAL e calculado DEPOIS, fora do cache, para responder a mudancas de ifr_inferior
            preco_atual = float(last_row['Close'])
            results.append({
                "Ticker": t.replace(".SA", ""),
                "Ticker_Full": t,
                "Preço": preco_atual,
                "IFR2": ifr_atual,
                "ATR": float(last_row['ATR']),
                "MM200": "✅ ACIMA" if last_row['Close'] > last_row['SMA200'] else "❌ ABAIXO",
                "WR @ Nível (3y)": f"{wr_nivel:.1f}% ({n_nivel}t)",
                "WR Setup (3y)": f"{wr_setup:.1f}% ({n_setup}t)",
                "Δ WR (pp)": f"{delta_wr:+.1f}",
                "Alvo": float(last_row['Alvo']),
                "Potencial %": ((float(last_row['Alvo']) / preco_atual) - 1) * 100,
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

# ============================================================
# SCORE SNIPER - Composicao institucional de convicção
# ============================================================
# Composicao (max 100 pts):
#   IFR Score        (40 pts) - quanto menor o IFR, maior o score
#   Delta WR Score   (30 pts) - premio de sobrevenda atual vs setup baseline
#   Fluxo B3 Score   (20 pts) - sell climax do estrangeiro = bonus
#   Liquidez Score   (10 pts) - ADTV em R$ MM
#
# Ver docs/SCORE_SNIPER.md para formula matematica detalhada
# ============================================================
def calcular_score_sniper(row, z_estrangeiro=0.0, ifr_limite=25):
    """
    Calcula Score Sniper (0-100) para uma linha do scanner.
    Regra critica: se IFR > ifr_limite (sem sinal de compra), score = 0.
    Ver docs/SCORE_SNIPER.md para formula completa.
    """
    # === Pre-condicao: sem sinal de compra -> score zero ===
    try:
        ifr = float(row.get("IFR2", 100))
    except (TypeError, ValueError):
        return 0.0
    if ifr > ifr_limite:
        return 0.0  # sem sinal operacional, score zero

    # === 1. IFR Score (45 pts max) - quanto menor o IFR, maior ===
    # Faixa: IFR=0 -> 45pts, IFR=25 -> 0pts (linear)
    ifr_score = 45 * (1 - ifr / ifr_limite)

    # === 2. Delta WR Score (25 pts max) - premio de sobrevenda ===
    try:
        delta_str = str(row.get("Δ WR (pp)", "0")).replace("pp", "").replace("+", "").strip()
        delta_wr = float(delta_str)
    except (TypeError, ValueError):
        delta_wr = 0.0
    # Mapeia delta_wr [-10, +15] -> score [0, 25]
    # delta=0 (neutro) -> 10pts; delta=+10 -> 20pts; delta=+15 -> 25pts
    delta_wr_score = max(0, min(25, 10 + delta_wr * 1.0))

    # === 3. Fluxo B3 Score (20 pts max) - sell climax do estrangeiro ===
    # Z = -2 -> 20pts (oportunidade extrema)
    # Z =  0 -> 10pts (neutro)
    # Z = +2 -> 0pts (estrangeiro ja comprou tudo)
    fluxo_score = max(0, min(20, 10 - z_estrangeiro * 5))

    # === 4. Liquidez Score (10 pts max) - ADTV em R$ MM ===
    try:
        vol_medio_m = float(row.get("Vol Médio (M)", 0))
        preco = float(row.get("Preço", 0))
        adtv = vol_medio_m * preco
    except (TypeError, ValueError):
        adtv = 0
    if   adtv >= 100: liq_score = 10
    elif adtv >=  50: liq_score = 8
    elif adtv >=  20: liq_score = 6
    elif adtv >=  10: liq_score = 4
    elif adtv >=   5: liq_score = 2
    else:             liq_score = 0

    total = ifr_score + delta_wr_score + fluxo_score + liq_score
    return round(min(total, 100), 1)


if botao_scan:
    with st.spinner('Escaneando mercado...'):
        lista_final = list(set(tickers_para_scan).union(st.session_state.tickers_adicionados))
        
        # Bucket de 2 minutos: invalida cache automatico durante o pregao
        import time
        ts_bucket = int(time.time() / 120)
        df_f, d_brutos = processar_dados_sniper(
            lista_final, ifr_gatilho_wr=ifr_inferior, _ts_bucket=ts_bucket
        )
        # Guarda timestamp do scan para exibir no header
        st.session_state.ultimo_scan_ts = time.time()
        # SINAL e calculado AQUI, fora do cache - responde a mudancas de ifr_inferior em tempo real
        df_f["SINAL"] = df_f["IFR2"].apply(
            lambda x: "🔥 COMPRA" if x < ifr_inferior else "AGUARDAR"
        )
        st.session_state.df_resultado = df_f
        st.session_state.dados_brutos = d_brutos

# Recalcula SINAL toda vez que a pagina renderiza (cobre mudanca de ifr_inferior sem novo scan)
if st.session_state.df_resultado is not None:
    st.session_state.df_resultado["SINAL"] = st.session_state.df_resultado["IFR2"].apply(
        lambda x: "🔥 COMPRA" if x < ifr_inferior else "AGUARDAR"
    )

    # === SCORE SNIPER (recalculado dinamicamente apos SCAN) ===
    # Pega Z-score do Estrangeiro do fluxo B3 ja cacheado
    z_estr_hoje = 0.0
    try:
        if "fluxo_b3_df" in st.session_state and st.session_state.fluxo_b3_df is not None:
            mf = calcular_metricas_fluxo(st.session_state.fluxo_b3_df)
            z_estr_hoje = mf.get("Estrangeiro", {}).get("z_score", 0.0)
        else:
            # Tenta puxar do cache sem bloquear UI (silent fail)
            try:
                df_fluxo_tmp = fetch_fluxo_b3(days=60)
                if not df_fluxo_tmp.empty:
                    st.session_state.fluxo_b3_df = df_fluxo_tmp
                    mf = calcular_metricas_fluxo(df_fluxo_tmp)
                    z_estr_hoje = mf.get("Estrangeiro", {}).get("z_score", 0.0)
            except Exception:
                pass
    except Exception:
        pass

    st.session_state.df_resultado["Score Sniper"] = st.session_state.df_resultado.apply(
        lambda r: calcular_score_sniper(r, z_estrangeiro=z_estr_hoje, ifr_limite=ifr_inferior), axis=1
    )
    # Ordena por Score descendente (maior convicao no topo)
    st.session_state.df_resultado = st.session_state.df_resultado.sort_values(
        "Score Sniper", ascending=False
    ).reset_index(drop=True)
    # Guarda z para exibicao informativa
    st.session_state.z_estr_hoje = z_estr_hoje

tab_mon, tab_back, tab_fluxo = st.tabs(["📊 Monitoramento", "🧪 Backtest por Ativo", "🌐 Fluxo B3"])

# --- MONITORAMENTO ---
with tab_mon:
    if st.session_state.df_resultado is not None:
        df_ex = st.session_state.df_resultado
        if 'Ticker_Full' not in df_ex.columns:
            st.warning("⚠️ Execute o SCAN.")
            st.stop()

        # === Header com timestamp do ultimo update + disclaimer yfinance ===
        import time
        ts_scan = st.session_state.get("ultimo_scan_ts", 0)
        if ts_scan:
            idade_seg = int(time.time() - ts_scan)
            if idade_seg < 60:
                idade_str = f"{idade_seg}s atras"
                cor_ts = CORES_SNIPER["verde_neon"]
            elif idade_seg < 300:
                idade_str = f"{idade_seg//60}min atras"
                cor_ts = CORES_SNIPER["azul_selecao"]
            else:
                idade_str = f"{idade_seg//60}min atras (DESATUALIZADO)"
                cor_ts = CORES_SNIPER["vermelho"]
            from datetime import datetime as _dt
            scan_dt = _dt.fromtimestamp(ts_scan).strftime("%H:%M:%S")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:8px;font-size:0.85em;'
                f'color:{CORES_SNIPER["text"]}">'
                f'<span>📡 Ultimo SCAN: <b style="color:{cor_ts}">{scan_dt}</b> '
                f'({idade_str})</span>'
                f'<span style="opacity:0.6">⚠️ Yahoo Finance: delay ~15min para B3</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Define a ordem para a nova coluna ficar ao lado do SINAL
        cols_base = ['Ticker', 'Score Sniper', 'Preço', 'IFR2', 'MM200', 'SINAL', 'WR @ Nível (3y)', 'WR Setup (3y)', 'Δ WR (pp)', 'Alvo', 'Potencial %', 'ATR', 'Vol Médio (M)', 'Vol_Hoje (M)', 'Vol_vs_Media', 'Data', 'Fluxo_OBV']
        cols_show = [c for c in cols_base if c in df_ex.columns]
        
        # Função para pintar o Score Sniper por faixa de conviccao
        def colorir_score(val):
            try:
                v = float(val)
                if v >= 85: return f'background-color: rgba(57,255,20,0.35); color: white; font-weight: bold'
                if v >= 70: return f'background-color: rgba(57,255,20,0.18); color: #39FF14; font-weight: bold'
                if v >= 50: return f'color: #FFD700; font-weight: bold'  # amarelo dourado
                if v >= 30: return f'color: {CORES_SNIPER["text"]}'
                return f'color: {CORES_SNIPER["vermelho"]}'
            except: return ''

        # Função para pintar o WR de Verde se >= 70% ou Vermelho se < 50%
        def colorir_wr(val):
            try:
                perc = float(val.split('%')[0])
                if perc >= 70.0: return f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold'
                elif perc < 50.0: return f'color: {CORES_SNIPER["vermelho"]}'
            except: pass
            return ''

        # === DEFESA CONTRA DataFrame ANTIGO (session_state pre-update) ===
        cols_wr_esperadas = ['WR @ Nível (3y)', 'WR Setup (3y)']
        if not all(c in df_ex.columns for c in cols_wr_esperadas):
            st.warning(
                "⚠️ DataFrame em cache esta desatualizado (versao anterior do codigo). "
                "Clique em **🧹 Limpar Cache** no sidebar e execute o SCAN novamente."
            )
            st.stop()

        # Subset dinamico: aplica colorir_wr apenas em colunas existentes
        wr_subset = [c for c in cols_wr_esperadas if c in df_ex.columns]
        sinal_subset = ['SINAL'] if 'SINAL' in df_ex.columns else []

        styled = df_ex[cols_show].style.format({
            "Preço": "R$ {:.2f}",
            "Alvo": "R$ {:.2f}",
            "IFR2": "{:.2f}",
            "Potencial %": "{:.2f}%",
            "Vol Médio (M)": "{:.2f}M"
        })
        if sinal_subset:
            styled = styled.map(
                lambda v: f'color: {CORES_SNIPER["verde_neon"]}; font-weight: bold' if v == "🔥 COMPRA" else '',
                subset=sinal_subset
            )
        if wr_subset:
            styled = styled.map(colorir_wr, subset=wr_subset)
        if 'Score Sniper' in df_ex.columns:
            styled = styled.map(colorir_score, subset=['Score Sniper'])
            styled = styled.format({'Score Sniper': '{:.1f}'}, subset=['Score Sniper'])

        st.dataframe(styled, use_container_width="stretch", hide_index=True)

        # Caption institucional do Score Sniper
        z_info = st.session_state.get("z_estr_hoje", 0.0)
        st.caption(
            f"🎯 **Score Sniper** (0-100): composto institucional. "
            f"IFR (40pts) + Δ WR (30pts) + Fluxo Estrangeiro (20pts) + Liquidez (10pts). "
            f"Tabela ordenada por Score. Z-score Estrangeiro hoje: **{z_info:+.2f}**. "
            f"Detalhes em [docs/SCORE_SNIPER.md](#)."
        )

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
            filtro_tendencia = st.selectbox(
                "Filtro de Tendência:",
                ["Sem Filtro", "SMA200", "SMA52"],
                index=0,
                help="Empirico do baseline 2021-2026: 'Sem Filtro' (Stormer puro) entrega ~10x mais Sharpe que SMA50/SMA200."
            )
            periodo_bt = st.selectbox("Simular nos últimos:", ["Todo o período (2 anos)", "12 meses", "6 meses", "3 meses"], index=0)
            
            st.markdown("---")
            st.write("🛡️ **Gerenciamento de Risco**")
            usar_stop_mm5 = st.checkbox(
                "Sair na MM5 (Se estiver no Lucro)", value=False,
                help="Variante do Stormer (saida antecipada). Setup puro usa apenas alvo+timestop."
            )
            ativar_stop_fixo = st.checkbox("Utilizar Stop Loss Fixo", value=False)
            perc_stop_bt = st.number_input("Distância do Stop (%)", value=5.0, disabled=not ativar_stop_fixo)
            
            usar_time_stop = st.checkbox("Usar Time Stop", value=True)
            time_stop_val = st.slider(
                "Dias Máx (Time Stop)", min_value=3, max_value=15, value=7,
                disabled=not usar_time_stop,
                help="7 candles e o time stop canonico Stormer (literatura QuantBrasil)."
            )

            st.markdown("---")
            st.write("💰 **Custos Transacionais**")
            aplicar_custos = st.checkbox(
                "Aplicar custos B3 reais (round trip)", value=False,
                help="Emolumentos B3 ~0,0325% + slippage estimado ~0,05% = 0,0825% por trade."
            )
            custo_trade_pct = 0.0825 if aplicar_custos else 0.0

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
                        res = (p_open / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE ALTA (ALVO)'})
                        em_operacao = False
                        continue
                    elif ativar_stop_fixo and p_open <= v_stop:
                        res = (p_open / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE BAIXA (STOP)'})
                        em_operacao = False
                        continue
                    elif usar_stop_mm5 and v_mm5 > p_entrada and p_open <= v_mm5:
                        res = (p_open / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'GAP DE BAIXA (STOP MM5)'})
                        em_operacao = False
                        continue

                    # B. EXECUÇÃO NO PREGÃO (Ordem Pessimista: Stop verificado ANTES do Alvo)
                    if ativar_stop_fixo and p_low <= v_stop:
                        res = (v_stop / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'STOP FIXO'})
                        em_operacao = False
                        continue
                        
                    elif usar_stop_mm5 and v_mm5 > p_entrada and p_low <= v_mm5:
                        res = (v_mm5 / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'STOP MM5'})
                        em_operacao = False
                        continue
                        
                    elif p_high >= v_alvo:
                        res = (v_alvo / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'ALVO ATINGIDO'})
                        em_operacao = False
                        continue

                    # C. TIME STOP
                    if usar_time_stop and dias_op >= time_stop_val:
                        res = (p_close / p_entrada) - 1 - custo_trade_pct/100
                        trades_bt.append({'Entrada': d_entrada, 'Saída': df_sim.index[i], 'Resultado %': res * 100, 'Status': 'TIME STOP'})
                        em_operacao = False
                        continue
                    
                    # D. ENCERRAMENTO FORÇADO (Fim do arquivo)
                    if i == len(df_sim) - 1:
                        res = (p_close / p_entrada) - 1 - custo_trade_pct/100
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
                
                # === METRICAS INSTITUCIONAIS (com Sharpe / Sortino / Calmar) ===
                # Daily returns para Sharpe/Sortino: usa retornos por trade como aproximacao
                import numpy as np
                rets_decimais = tdf['Resultado %'].values / 100.0
                if len(rets_decimais) > 1 and rets_decimais.std() > 0:
                    sharpe = (rets_decimais.mean() / rets_decimais.std()) * np.sqrt(252 / max(tdf['Acumulado Multiplicador'].size, 1))
                else:
                    sharpe = 0.0
                downside = rets_decimais[rets_decimais < 0]
                if len(downside) > 1 and downside.std() > 0:
                    sortino = (rets_decimais.mean() / downside.std()) * np.sqrt(252 / max(tdf['Acumulado Multiplicador'].size, 1))
                else:
                    sortino = 0.0
                # Calmar = CAGR aprox / |Max DD|
                dias_janela = (tdf['Saída'].iloc[-1] - tdf['Entrada'].iloc[0]).days if len(tdf) > 1 else 1
                anos = max(dias_janela / 365.25, 1/252)
                cagr = ((1 + total_ret/100) ** (1/anos) - 1) * 100 if total_ret > -100 else -100
                calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

                # Linha 1: metricas principais
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Retorno Acumulado", f"{total_ret:.2f}%")
                m2.metric("CAGR", f"{cagr:.2f}%")
                m3.metric("Max Drawdown", f"{max_dd:.2f}%")
                m4.metric("Total Trades", len(tdf))

                # Linha 2: metricas risk-adjusted institucionais
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Sharpe Ratio", f"{sharpe:.2f}",
                          help="Risk-adjusted return. >1.0 e bom, >2.0 e excelente.")
                m6.metric("Sortino Ratio", f"{sortino:.2f}",
                          help="Como Sharpe mas considera so volatilidade negativa.")
                m7.metric("Calmar Ratio", f"{calmar:.2f}",
                          help="CAGR / |Max DD|. Penaliza drawdown agudo.")
                m8.metric("Payoff", f"{payoff:.2f}",
                          help="Ganho medio / Prejuizo medio. Stormer e baixo (~0.6) com WR alto.")

                # Linha 3: WR + Avg Trade
                m9, m10, m11, m12 = st.columns(4)
                m9.metric("Taxa de Acerto", f"{win_rate:.1f}%")
                m10.metric("Avg. Trade", f"{avg_trade:.2f}%")
                profit_factor = abs(tdf[tdf['Resultado %']>0]['Resultado %'].sum() /
                                    tdf[tdf['Resultado %']<0]['Resultado %'].sum()) if (tdf['Resultado %']<0).any() else float('inf')
                m11.metric("Profit Factor", f"{profit_factor:.2f}",
                          help="Soma dos ganhos / Soma das perdas. >1.3 e bom.")
                if aplicar_custos:
                    m12.metric("Custo aplicado", f"-{custo_trade_pct:.4f}%/trade")
                else:
                    m12.metric("Modo", "TEORICO (s/ custos)")
                
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

# --- FLUXO B3 (DADOS OFICIAIS POR TIPO DE INVESTIDOR) ---
with tab_fluxo:
    st.subheader("🌐 Fluxo de Investidores na B3 — Dados Oficiais")
    st.caption(
        "Fonte: dadosdemercado.com.br/fluxo (baseado no Resumo Diário B3 por tipo "
        "de investidor). Atualização T+1 após fechamento. Valores em R$ milhões — "
        "positivo = compra líquida no dia."
    )

    if st.button("🔄 Atualizar Dados B3", key="btn_fluxo_b3"):
        st.cache_data.clear()
        st.session_state.pop("fluxo_b3_df", None)

    # Cache de sessao para nao refazer scraping a cada interacao
    if "fluxo_b3_df" not in st.session_state:
        with st.spinner("Buscando dados oficiais de fluxo B3..."):
            try:
                st.session_state.fluxo_b3_df = fetch_fluxo_b3(days=180)
            except Exception as e:
                st.error(f"Falha ao buscar fluxo B3: {e}")
                st.session_state.fluxo_b3_df = pd.DataFrame()

    df_fluxo = st.session_state.fluxo_b3_df

    if df_fluxo is None or df_fluxo.empty:
        st.warning(
            "⚠️ Não foi possível carregar os dados de fluxo da B3. "
            "Verifique sua conexão ou tente novamente em alguns minutos."
        )
    else:
        # === Metricas institucionais ===
        metricas = calcular_metricas_fluxo(df_fluxo)
        alertas = gerar_alertas_institucionais(metricas)
        data_ref = metricas.get("data_referencia", "N/D")

        st.markdown(f"**Data de referência:** `{data_ref}` "
                    f"| Dados disponíveis: **{len(df_fluxo)}** dias")

        # === Header com 5 metricas por categoria (R$ MM hoje + Z-score) ===
        st.markdown("### 📊 Fluxo do Dia por Tipo de Investidor")
        cols_cat = st.columns(5)
        categorias = [
            ("Estrangeiro", "🌍"),
            ("Institucional", "🏛️"),
            ("Pessoa_Fisica", "👤"),
            ("Inst_Financeira", "🏦"),
            ("Outros", "📦"),
        ]
        for i, (cat, emoji) in enumerate(categorias):
            with cols_cat[i]:
                m = metricas.get(cat, {})
                hoje = m.get("hoje", 0)
                z = m.get("z_score", 0)
                label = cat.replace("_", " ")
                # Cor por sinal
                delta_color = "normal"
                if m.get("sell_climax"):
                    delta_color = "inverse"
                cols_cat[i].metric(
                    f"{emoji} {label}",
                    f"R$ {hoje:+,.0f}mi",
                    f"Z={z:+.2f}"
                )

        st.divider()

        # === Alertas institucionais ===
        st.markdown("### 🚨 Alertas Institucionais")
        for alerta in alertas:
            tipo = alerta["tipo"]
            msg = alerta["msg"]
            if "OPORTUNIDADE" in tipo or "COMPRADOR" in tipo:
                st.success(f"**{tipo}** — {msg}")
            elif "EXAUSTAO" in tipo or "VENDEDOR" in tipo:
                st.warning(f"**{tipo}** — {msg}")
            else:
                st.info(f"**{tipo}** — {msg}")

        st.divider()

        # === Grafico de barras temporal (Estrangeiro + MM7) ===
        st.markdown("### 📈 Histórico do Fluxo Estrangeiro (90 dias)")
        df_plot = df_fluxo.tail(90).copy()
        df_plot["MM_7d"] = df_plot["Estrangeiro"].rolling(7).mean()

        cores = [CORES_SNIPER['verde_neon'] if v >= 0 else CORES_SNIPER['vermelho']
                 for v in df_plot["Estrangeiro"]]

        fig_fluxo = go.Figure()
        fig_fluxo.add_trace(go.Bar(
            x=df_plot["Data"], y=df_plot["Estrangeiro"],
            marker_color=cores, name="Fluxo Estrangeiro (R$ MM)",
            opacity=0.85
        ))
        fig_fluxo.add_trace(go.Scatter(
            x=df_plot["Data"], y=df_plot["MM_7d"],
            line=dict(color=CORES_SNIPER['azul_selecao'], width=2),
            name="Média Móvel 7d"
        ))
        fig_fluxo.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
        fig_fluxo.update_layout(
            template="plotly_dark", height=400,
            plot_bgcolor=CORES_SNIPER['bg_deep'],
            paper_bgcolor=CORES_SNIPER['bg_deep'],
            hovermode="x unified",
            yaxis_title="R$ Milhões",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Remove sab/dom para nao exibir barras vazias
        fig_fluxo.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig_fluxo, use_container_width=True)

        # === HEATMAP CALENDARIO (estilo GitHub) - regime de fluxo do ano ===
        st.markdown("### 🗓️ Heatmap do Fluxo — Estrangeiro · Este Ano")
        st.caption("Cada quadrado = 1 pregão · Cor = intensidade do fluxo "
                   "(venda forte → compra forte)")

        ano_atual = pd.Timestamp.today().year
        mat = construir_matriz_calendario(df_fluxo, "Estrangeiro", ano=ano_atual)
        stats_cal = estatisticas_calendario(df_fluxo, "Estrangeiro", ano=ano_atual)

        if mat and "z_matrix" in mat:
            col_heat, col_stats = st.columns([3, 1])

            with col_heat:
                # Escala divergente: vermelho forte -> cinza neutro -> verde forte
                colorscale_fluxo = [
                    [0.00, "#7A0000"],   # venda muito forte
                    [0.25, "#D90429"],   # venda forte
                    [0.45, "#3A0000"],   # venda fraca
                    [0.50, "#161B22"],   # neutro (zero)
                    [0.55, "#003500"],   # compra fraca
                    [0.75, "#39FF14"],   # compra forte
                    [1.00, "#00B800"],   # compra muito forte
                ]
                fig_heat = go.Figure(data=go.Heatmap(
                    z=mat["z_norm"],
                    x=mat["x_dates"],
                    y=mat["y_labels"],
                    text=mat["z_text"],
                    hoverinfo="text",
                    colorscale=colorscale_fluxo,
                    zmin=-1.0, zmax=1.0,
                    xgap=2, ygap=2,
                    showscale=False,
                ))
                # Configura xticks por mes
                tick_dates = list(mat["month_ticks"].keys())
                tick_labels = list(mat["month_ticks"].values())
                fig_heat.update_layout(
                    template="plotly_dark",
                    height=260,
                    margin=dict(l=40, r=10, t=20, b=30),
                    plot_bgcolor=CORES_SNIPER["bg_deep"],
                    paper_bgcolor=CORES_SNIPER["bg_deep"],
                    xaxis=dict(
                        tickmode="array",
                        tickvals=tick_dates,
                        ticktext=tick_labels,
                        showgrid=False,
                        tickfont=dict(size=11)
                    ),
                    yaxis=dict(
                        showgrid=False,
                        autorange="reversed",
                        tickfont=dict(size=11)
                    ),
                )
                # Legenda manual abaixo (chips de cor)
                st.markdown(
                    f'<div style="display:flex;gap:6px;align-items:center;'
                    f'margin-bottom:8px;font-size:0.85em;color:{CORES_SNIPER["text"]}">'
                    f'<span style="color:{CORES_SNIPER["vermelho"]};font-weight:bold">'
                    f'VENDA FORTE</span>'
                    f'<span style="background:#7A0000;width:18px;height:14px;'
                    f'display:inline-block;border-radius:2px"></span>'
                    f'<span style="background:#D90429;width:18px;height:14px;'
                    f'display:inline-block;border-radius:2px"></span>'
                    f'<span style="background:{CORES_SNIPER["cinza_grid"]};'
                    f'width:18px;height:14px;display:inline-block;border-radius:2px"></span>'
                    f'<span style="background:#39FF14;width:18px;height:14px;'
                    f'display:inline-block;border-radius:2px"></span>'
                    f'<span style="background:#00B800;width:18px;height:14px;'
                    f'display:inline-block;border-radius:2px"></span>'
                    f'<span style="color:{CORES_SNIPER["verde_neon"]};font-weight:bold">'
                    f'COMPRA FORTE</span></div>',
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            with col_stats:
                st.markdown("&nbsp;", unsafe_allow_html=True)  # espacamento
                # Estatisticas no estilo da imagem de referencia
                if stats_cal:
                    st.markdown(
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px;margin-top:4px">'
                        f'PREGÕES COM ENTRADA</div>'
                        f'<div style="color:{CORES_SNIPER["verde_neon"]};'
                        f'font-size:1.4em;font-weight:bold">'
                        f'{stats_cal["pregoes_compra"]} de {stats_cal["total_pregoes"]} '
                        f'({stats_cal["pct_compra"]:.0f}%)</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px;margin-top:12px">'
                        f'PREGÕES COM SAÍDA</div>'
                        f'<div style="color:{CORES_SNIPER["vermelho"]};'
                        f'font-size:1.4em;font-weight:bold">'
                        f'{stats_cal["pregoes_venda"]} de {stats_cal["total_pregoes"]} '
                        f'({stats_cal["pct_venda"]:.0f}%)</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px;margin-top:12px">'
                        f'MAIOR SEQUÊNCIA DE COMPRA</div>'
                        f'<div style="color:{CORES_SNIPER["verde_neon"]};'
                        f'font-size:1.2em;font-weight:bold">'
                        f'{stats_cal["maior_sequencia_compra"]} dias em '
                        f'{stats_cal["mes_run_compra"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px;margin-top:12px">'
                        f'MAIOR SEQUÊNCIA DE VENDA</div>'
                        f'<div style="color:{CORES_SNIPER["vermelho"]};'
                        f'font-size:1.2em;font-weight:bold">'
                        f'{stats_cal["maior_sequencia_venda"]} dias em '
                        f'{stats_cal["mes_run_venda"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="border:2px solid {CORES_SNIPER["laranja_mm200"]};'
                        f'border-radius:8px;padding:10px;margin-top:14px">'
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px">MÊS MAIS VERDE</div>'
                        f'<div style="color:{CORES_SNIPER["verde_neon"]};'
                        f'font-size:1.4em;font-weight:bold">'
                        f'{stats_cal["mes_mais_verde"]}</div>'
                        f'<div style="color:{CORES_SNIPER["verde_neon"]};opacity:0.7;'
                        f'font-size:0.85em">+R$ {stats_cal["valor_mes_verde"]:,.0f} mi</div>'
                        f'<div style="color:{CORES_SNIPER["text"]};opacity:0.7;'
                        f'font-size:0.75em;letter-spacing:0.5px;margin-top:10px">'
                        f'MÊS MAIS VERMELHO</div>'
                        f'<div style="color:{CORES_SNIPER["vermelho"]};'
                        f'font-size:1.4em;font-weight:bold">'
                        f'{stats_cal["mes_mais_vermelho"]}</div>'
                        f'<div style="color:{CORES_SNIPER["vermelho"]};opacity:0.7;'
                        f'font-size:0.85em">R$ {stats_cal["valor_mes_vermelho"]:,.0f} mi</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("Dados insuficientes para gerar o heatmap calendário do ano corrente.")

        # === Grafico empilhado: contribuicao de cada categoria ===
        st.markdown("### 🔄 Composição do Fluxo (últimos 30 dias)")
        df_stack = df_fluxo.tail(30).copy()
        fig_stack = go.Figure()
        cores_cat = {
            "Estrangeiro": CORES_SNIPER['laranja_mm200'],
            "Institucional": CORES_SNIPER['azul_selecao'],
            "Pessoa_Fisica": CORES_SNIPER['verde_neon'],
            "Inst_Financeira": CORES_SNIPER['roxo_mm52'],
            "Outros": CORES_SNIPER['cinza_grid'],
        }
        for cat, cor in cores_cat.items():
            if cat in df_stack.columns:
                fig_stack.add_trace(go.Bar(
                    x=df_stack["Data"], y=df_stack[cat],
                    name=cat.replace("_", " "),
                    marker_color=cor, opacity=0.85
                ))
        fig_stack.update_layout(
            barmode="relative",
            template="plotly_dark", height=400,
            plot_bgcolor=CORES_SNIPER['bg_deep'],
            paper_bgcolor=CORES_SNIPER['bg_deep'],
            yaxis_title="R$ Milhões",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Remove sab/dom para nao exibir barras vazias
        fig_stack.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig_stack, use_container_width=True)

        # === Tabela detalhada dos ultimos 30 dias ===
        with st.expander("📋 Tabela detalhada (30 dias)"):
            df_tab = df_fluxo.tail(30).copy().iloc[::-1]
            df_tab["Data"] = df_tab["Data"].dt.strftime("%d/%m/%Y")
            cols_tab = ["Data", "Estrangeiro", "Institucional", "Pessoa_Fisica",
                        "Inst_Financeira", "Outros"]
            cols_tab = [c for c in cols_tab if c in df_tab.columns]
            st.dataframe(
                df_tab[cols_tab].style.format({
                    c: "R$ {:+,.0f}mi" for c in cols_tab if c != "Data"
                }).map(
                    lambda v: f"color: {CORES_SNIPER['verde_neon']}" if isinstance(v, (int, float)) and v > 0
                    else (f"color: {CORES_SNIPER['vermelho']}" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=[c for c in cols_tab if c != "Data"]
                ),
                use_container_width=True, hide_index=True
            )

        st.caption(
            "💡 **Interpretação Sniper Quant:** Quando o **Estrangeiro entra em "
            "Sell Climax** (Z < -2σ) e há sinais IFR2 ≤ 25 simultâneos no mercado, "
            "há **dupla confluência** de mean reversion — sobreponderar entradas. "
            "Em regime vendedor persistente do estrangeiro, **reduzir size** mesmo "
            "com sinal técnico válido."
        )
