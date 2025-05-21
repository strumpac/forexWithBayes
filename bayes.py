import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

max_distanza_pips = 20

def distanza_pips(entry, prezzo_attuale):
    if entry > 50:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    return abs(prezzo_attuale - entry) / pip_size

def expand_to_daily(df, start_date, end_date):
    df_daily = df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
    df_daily = df_daily.ffill()
    return df_daily

def calcola_profitto_perdita(entry, tp, sl, lotti, prezzo_per_pip=10):
    if entry > 50:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    pips_tp = abs(tp - entry) / pip_size
    pips_sl = abs(entry - sl) / pip_size
    profit = pips_tp * prezzo_per_pip * lotti
    loss = pips_sl * prezzo_per_pip * lotti
    return profit, loss

def stima_tempo_chiusura(df, forex_name, entry, tp, sl, signal):
    prezzi = df[forex_name].dropna()
    for i in range(1, len(prezzi)):
        prezzo = prezzi.iloc[-i]
        if signal == 'LONG':
            if prezzo >= tp:
                return i, 'TP'
            elif prezzo <= sl:
                return i, 'SL'
        elif signal == 'SHORT':
            if prezzo <= tp:
                return i, 'TP'
            elif prezzo >= sl:
                return i, 'SL'
    return None, None

def bayes_signal_with_levels(df, forex_col, macro_conditions, risk_level):
    df_cond = df.copy()
    for col_macro, val_macro in macro_conditions.items():
        df_cond = df_cond[df_cond[col_macro] == val_macro]
    if len(df_cond) == 0:
        return None, 0, 0, None, None, None

    p_up = df_cond[forex_col].mean()
    p_down = 1 - p_up
    signal = 'LONG' if p_up > 0.5 else 'SHORT' if p_down > 0.5 else 'NEUTRAL'
    current_price = df[forex_col.replace('_up', '')].iloc[-1]
    entry = current_price

    if risk_level == 1:
        tp_factor = 1.005
        sl_factor = 0.998
    elif risk_level == 2:
        tp_factor = 1.01
        sl_factor = 0.995
    else:
        tp_factor = 1.02
        sl_factor = 0.99

    if signal == 'LONG':
        tp = entry * tp_factor
        sl = entry * sl_factor
    elif signal == 'SHORT':
        tp = entry * (2 - tp_factor)
        sl = entry * (2 - sl_factor)
    else:
        tp = sl = None

    return signal, p_up, p_down, entry, tp, sl

# Nuova funzione per determinare tipo ordine
def determina_tipo_ordine(signal, entry, current_price):
    if signal == 'LONG':
        if entry < current_price:
            return 'BUY LIMIT'
        else:
            return 'MARKET BUY'
    elif signal == 'SHORT':
        if entry > current_price:
            return 'SELL LIMIT'
        else:
            return 'MARKET SELL'
    return 'NONE'

# --- Scarica dati ---
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

coppie = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'CAD=X',
    'NZDUSD': 'NZDUSD=X',
    'EURGBP': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'CHFJPY': 'CHFJPY=X',
    'CADCHF': 'CADCHF=X'
}

forex_data = {}
for name, ticker in coppie.items():
    df_fx = yf.download(ticker, start=start, end=end)['Close']
    forex_data[name] = df_fx

macro_symbols = {
    'GDP': 'GDP',
    'UNRATE': 'UNRATE',
    'CPI': 'CPIAUCSL',
    'FEDFUNDS': 'FEDFUNDS'
}

macro_data = {}
for name, symbol in macro_symbols.items():
    df_macro = pdr.DataReader(symbol, 'fred', start, end).dropna()
    macro_data[name] = df_macro

for name in macro_data:
    macro_data[name] = expand_to_daily(macro_data[name], start, end)

df = pd.DataFrame(index=pd.date_range(start, end, freq='D'))
for name in coppie:
    df[name] = forex_data[name].reindex(df.index).ffill()

for name in macro_data:
    df[name] = macro_data[name].reindex(df.index).ffill()

for name in coppie:
    df[name+'_up'] = (df[name].pct_change() > 0).astype(int)

df['GDP_change'] = df['GDP'].pct_change(periods=90)
df['GDP_up'] = (df['GDP_change'] > 0).astype(int)

df['UNRATE_change'] = df['UNRATE'].pct_change(periods=30)
df['UNRATE_down'] = (df['UNRATE_change'] < 0).astype(int)

df['CPI_change'] = df['CPI'].pct_change(periods=30)
df['CPI_up'] = (df['CPI_change'] > 0).astype(int)

df['FEDFUNDS_change'] = df['FEDFUNDS'].pct_change(periods=30)
df['FEDFUNDS_up'] = (df['FEDFUNDS_change'] > 0).astype(int)

df.dropna(inplace=True)

# --- Condizioni macro positive ---
macro_pos_conditions = {
    'GDP_up': 1,
    'UNRATE_down': 1,
    'CPI_up': 1,
    'FEDFUNDS_up': 1
}

# --- Impostazioni ---
budget = 151.0
lotti = 0.1
risk_level = 2

# --- Esecuzione finale ---
print("\nSegnali Forex con tipo ordine, livelli, stima tempo e consiglio:\n")

for name in coppie:
    forex_col = name + '_up'
    result = bayes_signal_with_levels(df, forex_col, macro_pos_conditions, risk_level)

    if result[0] is None:
        print(f"{name}: dati insufficienti per calcolo segnale.\n")
        continue

    signal, p_up, p_down, entry, tp, sl = result
    try:
      current_price = yf.Ticker(coppie[name]).info['regularMarketPrice']
    except:
      current_price = df[name].iloc[-1]
      print(f"⚠️ Prezzo RT non disponibile per {name}, uso ultimo dato storico.")


    dist_pips = distanza_pips(entry, current_price)
    if dist_pips > max_distanza_pips:
      print(f"{name} - ❌ Operazione scartata: distanza di {dist_pips:.1f} pips dall'entry (>{max_distanza_pips}).")
      continue


    ordine = determina_tipo_ordine(signal, entry, current_price)

    tp_str = f"{tp:.5f}" if tp else "N/A"
    sl_str = f"{sl:.5f}" if sl else "N/A"

    print(f"{name}: Segnale = {signal}, Ordine = {ordine}")
    print(f"Prob. salita = {p_up:.2f}, discesa = {p_down:.2f}")
    print(f"Entry: {entry:.5f}, TP: {tp_str}, SL: {sl_str}")

    if signal == 'NEUTRAL':
        print(f"{name} - Nessun segnale operativo.\n")
        continue

    profit, loss = calcola_profitto_perdita(entry, tp, sl, lotti)
    print(f"Profitto potenziale: ${profit:.2f}, Perdita potenziale: ${loss:.2f}")

    giorni_tp, chiusura_tp = stima_tempo_chiusura(df, name, entry, tp, sl, signal)
    giorni_sl, chiusura_sl = stima_tempo_chiusura(df, name, entry, tp, sl, signal)

    if chiusura_tp == 'TP':
        giorni_stimati = giorni_tp
        chiusura = 'TP'
    elif chiusura_sl == 'SL':
        giorni_stimati = giorni_sl
        chiusura = 'SL'
    else:
        giorni_stimati = None
        chiusura = None

    if giorni_stimati:
        print(f"Stimati {giorni_stimati} giorni per chiusura ({chiusura})")

        if chiusura == 'TP':
            print(f"✅ Consigliato: TP stimato prima dello SL.")
        elif chiusura == 'SL':
            if loss > budget:
                print(f"❌ Sconsigliato: SL stimato e perdita (${loss:.2f}) > budget (${budget:.2f})")
            else:
                print(f"⚠️ Attenzione: SL stimato ma perdita entro il budget.")
    else:
        print("Nessuna chiusura stimata nei dati storici.")

    print()
    #in questo programma capita spesso che dia un attenzione sl stimato ma perdita entro il budget. Vorrei che gestosse diversamente il tp e sl in modo che o sconsiglia l’acquisto o lo consiglia. Bada che non devi solo eliminare quella voce ma gestire bene tp e sl