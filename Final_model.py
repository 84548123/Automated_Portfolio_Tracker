# advisor_live.py (Upgraded with a Combined Signal Model)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import warnings
import os

# Suppress TensorFlow warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Data Fetching ---
def fetch_live_data(ticker, days=500):
    """Fetches, validates, and cleans the last N days of closing prices."""
    print(f"  -> Fetching latest price data for {ticker}...")
    try:
        data = yf.download(ticker, period=f"{days+100}d", interval="1d", progress=False)
        if data.empty: raise ValueError(f"No data returned for ticker '{ticker}'.")
        if 'Close' not in data.columns: raise ValueError(f"Data for '{ticker}' lacks 'Close' column.")
        close_prices = data['Close'].dropna()
        if isinstance(close_prices, pd.DataFrame): close_prices = close_prices.iloc[:, 0]
        return close_prices.tolist()
    except Exception as e:
        raise ConnectionError(f"Failed to process price data for {ticker}. Error: {e}")

def fetch_fundamental_data(ticker):
    """Fetches the P/E ratio for a given ticker."""
    print(f"  -> Fetching fundamental data for {ticker}...")
    try:
        info = yf.Ticker(ticker).info
        pe_ratio = info.get('trailingPE', info.get('forwardPE')) # Use forward PE as a fallback
        if pe_ratio is None:
            raise ValueError(f"P/E ratio not available for {ticker}.")
        return {"pe_ratio": pe_ratio}
    except Exception as e:
         raise ConnectionError(f"Failed to fetch fundamental data for {ticker}. Error: {e}")


# --- Model 1: Technical Indicators ---
def analyze_with_indicators(prices, name):
    # ... (code remains the same)
    sma50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else 0
    sma200 = sum(prices[-200:]) / 200 if len(prices) >= 200 else 0
    trend = "Neutral"
    if sma50 > sma200 * 1.01: trend = "Bullish"
    if sma50 < sma200 * 0.99: trend = "Bearish"
    
    series = pd.Series(prices)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean().iloc[-1]
    rs = gain / loss if loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    
    momentum = "Neutral"
    if rsi > 70: momentum = "Overbought"
    if rsi < 30: momentum = "Oversold"
    
    return {"name": name, "type": "Indicators", "trend": trend, "momentum": momentum, "rsi": f"{rsi:.2f}"}

def determine_allocation_from_indicators(nifty, smallcap, risk):
    # ... (code remains the same)
    base_nifty, base_smallcap = {'conservative': (0.7,0.3), 'balanced': (0.5,0.5), 'aggressive': (0.3,0.7)}.get(risk, (0.5,0.5))
    adjustment = 0.0
    if nifty["trend"] == "Bullish" and smallcap["trend"] != "Bullish": adjustment += 0.15
    if smallcap["trend"] == "Bullish" and nifty["trend"] != "Bullish": adjustment -= 0.15
    final_nifty = max(0.05, min(0.95, base_nifty + adjustment))
    return {"nifty": final_nifty, "smallcap": 1 - final_nifty}

# --- Model 2: LSTM (Advanced) ---
def analyze_with_lstm(prices, name):
    # ... (code remains the same)
    print(f"  -> Training live LSTM for {name}... (This will take a few minutes)")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    sequence_length = 60
    X_train, y_train = [scaled_data[i-sequence_length:i, 0] for i in range(sequence_length, len(scaled_data))], [scaled_data[i, 0] for i in range(sequence_length, len(scaled_data))]
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)), LSTM(32, return_sequences=False), Dense(16), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    predictions_scaled = []
    for _ in range(10):
        next_pred = model.predict(last_sequence, verbose=0)
        predictions_scaled.append(next_pred[0][0])
        last_sequence = np.append(last_sequence[0][1:], next_pred).reshape(1, sequence_length, 1)
    predicted_price = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1,1))[-1][0]
    predicted_return = (predicted_price - prices[-1]) / prices[-1]
    return {"name": name, "type": "LSTM", "predicted_return": f"{predicted_return:.2%}"}

def determine_allocation_from_lstm(nifty, smallcap, risk):
    # ... (code remains the same)
    base_nifty, base_smallcap = {'conservative': (0.7,0.3), 'balanced': (0.5,0.5), 'aggressive': (0.3,0.7)}.get(risk, (0.5,0.5))
    nifty_return = float(nifty['predicted_return'].strip('%')) / 100
    smallcap_return = float(smallcap['predicted_return'].strip('%')) / 100
    return_diff = nifty_return - smallcap_return
    adjustment = return_diff * 2.5
    final_nifty = max(0.05, min(0.95, base_nifty + adjustment))
    return {"nifty": final_nifty, "smallcap": 1 - final_nifty}
    
# --- Model 3: Fundamental Analysis ---
def analyze_with_fundamentals(nifty_pe_data, name):
    """Analyzes the fetched Nifty 50 P/E ratio from a reliable ETF source."""
    pe_ratio = nifty_pe_data['pe_ratio']
    
    valuation = "Neutral"
    if pe_ratio > 25: valuation = "High (Expensive)"
    if pe_ratio < 18: valuation = "Low (Cheap)"

    return {"name": name, "type": "Fundamentals", "pe_ratio": pe_ratio, "valuation": valuation}

def determine_allocation_from_fundamentals(nifty_analysis, risk):
    """Determines allocation based on Nifty 50's valuation."""
    base_nifty, base_smallcap = {'conservative': (0.7,0.3), 'balanced': (0.5,0.5), 'aggressive': (0.3,0.7)}.get(risk, (0.5,0.5))
    
    adjustment = 0.0
    if nifty_analysis['valuation'] == "High (Expensive)": adjustment = -0.20 
    if nifty_analysis['valuation'] == "Low (Cheap)": adjustment = 0.20
    
    final_nifty = max(0.05, min(0.95, base_nifty + adjustment))
    return {"nifty": final_nifty, "smallcap": 1 - final_nifty}

# --- Model 4: Combined Signal Model (NEW) ---
def determine_allocation_from_combined(tech_nifty, tech_smallcap, lstm_nifty, lstm_smallcap, fund_nifty, risk):
    """Determines allocation based on a weighted score from all three models."""
    base_nifty, base_smallcap = {'conservative': (0.7,0.3), 'balanced': (0.5,0.5), 'aggressive': (0.3,0.7)}.get(risk, (0.5,0.5))
    
    # Calculate a conviction score for Nifty 50
    nifty_score = 0
    
    # 1. Technical Signal (Weight: 1.0)
    if tech_nifty['trend'] == 'Bullish': nifty_score += 1.0
    if tech_nifty['trend'] == 'Bearish': nifty_score -= 1.0
        
    # 2. LSTM Signal (Weight: 1.5 - higher weight as it's predictive)
    lstm_nifty_return = float(lstm_nifty['predicted_return'].strip('%'))
    lstm_smallcap_return = float(lstm_smallcap['predicted_return'].strip('%'))
    if lstm_nifty_return > lstm_smallcap_return: nifty_score += 1.5
    else: nifty_score -= 1.5
        
    # 3. Fundamental Signal (Weight: 1.0)
    if fund_nifty['valuation'] == 'Low (Cheap)': nifty_score += 1.0
    if fund_nifty['valuation'] == 'High (Expensive)': nifty_score -= 1.0
        
    # Max score is 3.5. Convert score to an adjustment percentage (max +/- 35%)
    adjustment = (nifty_score / 3.5) * 0.35
    
    final_nifty = max(0.05, min(0.95, base_nifty + adjustment))
    return {"nifty": final_nifty, "smallcap": 1 - final_nifty}

# --- UI and Display ---
def display_results(allocation, analyses):
    nifty_pct, smallcap_pct = allocation['nifty'] * 100, allocation['smallcap'] * 100
    
    print("\n" + "="*50)
    print("      LIVE PORTFOLIO ALLOCATION ADVISOR")
    print("="*50)
    
    print(f"\n--- Market Analysis (using {analyses['nifty_tech']['type']} Model) ---")
    
    if 'nifty_fund' in analyses: # Combined Model Output
        print("  --- Individual Model Signals ---")
        print(f"  - Technical Trend Signal:    {analyses['nifty_tech']['trend']}")
        lstm_winner = "Nifty 50" if float(analyses['nifty_lstm']['predicted_return'].strip('%')) > float(analyses['smallcap_lstm']['predicted_return'].strip('%')) else "Smallcap 250"
        print(f"  - LSTM Prediction Signal:    Favors {lstm_winner}")
        print(f"  - Fundamental Signal:        Market Valuation is {analyses['nifty_fund']['valuation']}")
    elif analyses['nifty_tech']['type'] == 'Indicators':
        print(f"  - Nifty 50:     Trend={analyses['nifty_tech']['trend']}, Momentum={analyses['nifty_tech']['momentum']} (RSI: {analyses['nifty_tech']['rsi']})")
        print(f"  - Smallcap 250: Trend={analyses['smallcap_tech']['trend']}, Momentum={analyses['smallcap_tech']['momentum']} (RSI: {analyses['smallcap_tech']['rsi']})")
    elif analyses['nifty_tech']['type'] == 'LSTM':
        print(f"  - Nifty 50 Predicted 10-Day Return:     {analyses['nifty_tech']['predicted_return']}")
        print(f"  - Smallcap 250 Predicted 10-Day Return: {analyses['smallcap_tech']['predicted_return']}")
    else: # Fundamentals
        print(f"  - {analyses['nifty_tech']['name']} P/E Ratio: {analyses['nifty_tech']['pe_ratio']:.2f}")
        print(f"  - Market Valuation:              {analyses['nifty_tech']['valuation']}")

    print("\n--- Suggested Allocation ---")
    bar = "#" * int(nifty_pct/2) + "-" * int(smallcap_pct/2)
    print(f"  [{bar}]")
    print(f"  Nifty 50:     {nifty_pct:.1f}%")
    print(f"  Smallcap 250: {smallcap_pct:.1f}%")
    
    print("\n--- How to Get Better Suggestions ---")
    print("  1. Refine Model Weights: Adjust the importance of each signal in the Combined Model.")
    print("  2. Expand Diversification: Include other asset classes like Gold or Bonds.")
    print("  3. Backtest the Strategy: Test how the Combined Model would have performed historically.")
    print("\nDisclaimer: This is an educational tool, not financial advice.")
    print("="*50)

# --- Main Program Execution ---
if __name__ == "__main__":
    try:
        print("Select Your Analysis Model:")
        print("  1. Technical Indicators (Fast, SMA/RSI)")
        print("  2. Advanced Model (Slow, predictive LSTM)")
        print("  3. Fundamental Analysis (Fast, Nifty 50 P/E Ratio)")
        print("  4. Combined Signal Model (Most Robust)")
        model_choice = input("Enter choice (1/2/3/4): ")
        
        print("\nSelect Your Risk Profile:")
        print("  1. Conservative")
        print("  2. Balanced")
        print("  3. Aggressive")
        risk_choice = input("Enter choice (1/2/3): ")
        risk_profile = {"1": "conservative", "2": "balanced", "3": "aggressive"}.get(risk_choice, "balanced")
        
        print(f"\nInitializing for '{risk_profile}' profile...")

        nifty_etf = 'NIFTYBEES.NS'
        smallcap_etf = 'HDFCSML250.NS'

        if model_choice == '4': # Combined Model
            # 1. Fetch all necessary data
            nifty_prices = fetch_live_data(nifty_etf)
            smallcap_prices = fetch_live_data(smallcap_etf)
            
            nifty_etf_tickers = {'NIFTYBEES.NS': 'Nifty 50 (NiftyBees ETF)', 'ICICINIFTY.NS': 'Nifty 50 (ICICI ETF)'}
            nifty_pe_data = None
            used_ticker_name = ""
            for ticker, name in nifty_etf_tickers.items():
                try:
                    nifty_pe_data = fetch_fundamental_data(ticker)
                    used_ticker_name = name
                    break
                except Exception:
                    print(f"  -> Note: Could not fetch fundamentals for {ticker}. Trying next...")
            if nifty_pe_data is None: raise ConnectionError("Could not fetch fundamental data.")

            # 2. Run all three analyses
            nifty_tech = analyze_with_indicators(nifty_prices, "Nifty 50 (ETF)")
            smallcap_tech = analyze_with_indicators(smallcap_prices, "Smallcap 250 (ETF)")
            nifty_lstm = analyze_with_lstm(nifty_prices, "Nifty 50 (ETF)")
            smallcap_lstm = analyze_with_lstm(smallcap_prices, "Smallcap 250 (ETF)")
            nifty_fund = analyze_with_fundamentals(nifty_pe_data, used_ticker_name)

            # 3. Determine allocation
            allocation = determine_allocation_from_combined(nifty_tech, smallcap_tech, nifty_lstm, smallcap_lstm, nifty_fund, risk_profile)
            
            # 4. Display results
            all_analyses = {
                "nifty_tech": nifty_tech, "smallcap_tech": smallcap_tech,
                "nifty_lstm": nifty_lstm, "smallcap_lstm": smallcap_lstm,
                "nifty_fund": nifty_fund
            }
            display_results(allocation, all_analyses)

        elif model_choice == '3': # Fundamentals
            nifty_etf_tickers = {'NIFTYBEES.NS': 'Nifty 50 (NiftyBees ETF)', 'ICICINIFTY.NS': 'Nifty 50 (ICICI ETF)'}
            nifty_pe_data = None
            used_ticker_name = ""
            for ticker, name in nifty_etf_tickers.items():
                try:
                    nifty_pe_data = fetch_fundamental_data(ticker)
                    used_ticker_name = name
                    break
                except Exception:
                    print(f"  -> Note: Could not fetch fundamentals for {ticker}. Trying next...")
            if nifty_pe_data is None: raise ConnectionError("Could not fetch fundamental data.")

            nifty_analysis = analyze_with_fundamentals(nifty_pe_data, used_ticker_name)
            allocation = determine_allocation_from_fundamentals(nifty_analysis, risk_profile)
            display_results(allocation, {"nifty_tech": nifty_analysis})

        else: # LSTM or Indicators
            nifty_prices = fetch_live_data(nifty_etf)
            smallcap_prices = fetch_live_data(smallcap_etf)
            
            if model_choice == '2': # LSTM
                nifty_analysis = analyze_with_lstm(nifty_prices, "Nifty 50 (ETF)")
                smallcap_analysis = analyze_with_lstm(smallcap_prices, "Smallcap 250 (ETF)")
                allocation = determine_allocation_from_lstm(nifty_analysis, smallcap_analysis, risk_profile)
            else: # Indicators
                nifty_analysis = analyze_with_indicators(nifty_prices, "Nifty 50 (ETF)")
                smallcap_analysis = analyze_with_indicators(smallcap_prices, "Smallcap 250 (ETF)")
                allocation = determine_allocation_from_indicators(nifty_analysis, smallcap_analysis, risk_profile)
            
            display_results(allocation, {"nifty_tech": nifty_analysis, "smallcap_tech": smallcap_analysis})

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


