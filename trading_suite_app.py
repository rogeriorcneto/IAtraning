import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import ccxt
import asyncio
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Trading Suite - 10 Aplicativos Integrados",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Classe principal da Trading Suite
class TradingSuite:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.scaler = MinMaxScaler()
        
    # 1. ANÁLISE DE SENTIMENTO
    def analyze_sentiment(self, text):
        """Análise de sentimento usando VADER e TextBlob"""
        vader_scores = self.analyzer.polarity_scores(text)
        blob = TextBlob(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def get_market_sentiment(self, symbol):
        """Coleta sentimento de mercado para um ativo"""
        try:
            # Simula coleta de notícias (normalmente usaria NewsAPI)
            news_samples = [
                f"{symbol} shows strong performance in recent trading sessions",
                f"Analysts remain bullish on {symbol} prospects",
                f"Market volatility affects {symbol} trading volumes"
            ]
            
            sentiments = []
            for news in news_samples:
                sentiment = self.analyze_sentiment(news)
                sentiments.append(sentiment)
            
            # Calcula sentimento médio
            avg_sentiment = {
                'compound': np.mean([s['vader_compound'] for s in sentiments]),
                'positive': np.mean([s['vader_positive'] for s in sentiments]),
                'negative': np.mean([s['vader_negative'] for s in sentiments])
            }
            
            return avg_sentiment, sentiments
        except Exception as e:
            st.error(f"Erro na análise de sentimento: {e}")
            return None, []
    
    # 2. ANÁLISE PREDITIVA
    def predict_prices(self, symbol, days=30):
        """Previsão de preços usando Random Forest"""
        try:
            # Coleta dados históricos
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            if data.empty:
                return None, None
            
            # Prepara features
            data['Returns'] = data['Close'].pct_change()
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
            
            # Remove NaN
            data = data.dropna()
            
            # Features para ML
            features = ['Returns', 'MA_5', 'MA_20', 'Volatility', 'Volume_MA']
            X = data[features].values
            y = data['Close'].values
            
            # Treina modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X[:-days], y[days:])
            
            # Faz previsões
            predictions = []
            last_features = X[-1].reshape(1, -1)
            
            for _ in range(days):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                # Atualiza features (simplificado)
                last_features[0, 0] = (pred - y[-1]) / y[-1]  # Returns simulado
            
            return predictions, data
        except Exception as e:
            st.error(f"Erro na previsão: {e}")
            return None, None
    
    # 3. TRADING DE ALTA FREQUÊNCIA (Simulado)
    def hft_scanner(self, symbols):
        """Scanner para oportunidades HFT"""
        opportunities = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if len(data) > 5:
                    # Calcula indicadores rápidos
                    current_price = data['Close'].iloc[-1]
                    sma_5 = data['Close'].rolling(5).mean().iloc[-1]
                    volume_spike = data['Volume'].iloc[-1] / data['Volume'].mean()
                    
                    # Detecta oportunidades
                    if current_price > sma_5 * 1.002 and volume_spike > 1.5:
                        opportunities.append({
                            'symbol': symbol,
                            'price': current_price,
                            'signal': 'BUY',
                            'strength': min(volume_spike, 5.0),
                            'timestamp': datetime.now()
                        })
                    elif current_price < sma_5 * 0.998 and volume_spike > 1.5:
                        opportunities.append({
                            'symbol': symbol,
                            'price': current_price,
                            'signal': 'SELL',
                            'strength': min(volume_spike, 5.0),
                            'timestamp': datetime.now()
                        })
            except:
                continue
        
        return opportunities
    
    # 4. ARBITRAGEM
    def find_arbitrage_opportunities(self, symbol):
        """Encontra oportunidades de arbitragem entre exchanges"""
        exchanges = ['binance', 'coinbasepro', 'kraken']
        prices = {}
        
        for exchange_name in exchanges:
            try:
                # Simula preços diferentes (normalmente usaria APIs reais)
                base_price = np.random.uniform(45000, 55000)  # Simula BTC
                spread = np.random.uniform(-0.002, 0.002)
                prices[exchange_name] = base_price * (1 + spread)
            except:
                continue
        
        if len(prices) >= 2:
            min_exchange = min(prices, key=prices.get)
            max_exchange = max(prices, key=prices.get)
            
            arbitrage_profit = (prices[max_exchange] - prices[min_exchange]) / prices[min_exchange]
            
            if arbitrage_profit > 0.001:  # 0.1% mínimo
                return {
                    'symbol': symbol,
                    'buy_exchange': min_exchange,
                    'sell_exchange': max_exchange,
                    'buy_price': prices[min_exchange],
                    'sell_price': prices[max_exchange],
                    'profit_pct': arbitrage_profit * 100,
                    'timestamp': datetime.now()
                }
        
        return None
    
    # 5. OTIMIZAÇÃO DE PORTFÓLIO
    def optimize_portfolio(self, symbols, investment_amount=10000):
        """Otimização de portfólio usando teoria moderna"""
        try:
            # Coleta dados
            data = yf.download(symbols, period="1y")['Close']
            
            if data.empty:
                return None
            
            # Calcula retornos
            returns = data.pct_change().dropna()
            
            # Calcula métricas
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Função objetivo (maximizar Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_vol  # Negative for minimization
            
            # Restrições
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(symbols)))
            
            # Otimização
            result = minimize(objective, np.array([1/len(symbols)] * len(symbols)),
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(mean_returns * optimal_weights)
                portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = portfolio_return / portfolio_vol
                
                # Calcula alocação em dólares
                allocations = {}
                for i, symbol in enumerate(symbols):
                    allocations[symbol] = {
                        'weight': optimal_weights[i],
                        'amount': optimal_weights[i] * investment_amount
                    }
                
                return {
                    'allocations': allocations,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio
                }
        except Exception as e:
            st.error(f"Erro na otimização: {e}")
            return None
    
    # 6. GERENCIAMENTO DE RISCO
    def risk_analysis(self, symbol, position_size=10000):
        """Análise de risco detalhada"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            if data.empty:
                return None
            
            # Calcula métricas de risco
            returns = data['Close'].pct_change().dropna()
            
            # VaR (Value at Risk)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # CVaR (Conditional VaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Volatilidade
            volatility = returns.std() * np.sqrt(252)
            
            # Calcula risco monetário
            risk_metrics = {
                'var_95_pct': var_95 * 100,
                'var_99_pct': var_99 * 100,
                'cvar_95_pct': cvar_95 * 100,
                'cvar_99_pct': cvar_99 * 100,
                'max_drawdown_pct': max_drawdown * 100,
                'volatility_pct': volatility * 100,
                'var_95_amount': abs(var_95 * position_size),
                'var_99_amount': abs(var_99 * position_size),
                'max_loss_estimate': abs(max_drawdown * position_size)
            }
            
            return risk_metrics
        except Exception as e:
            st.error(f"Erro na análise de risco: {e}")
            return None
    
    # 7. BOT DE TRADING (Simulado)
    def generate_trading_signals(self, symbol, strategy='sma_crossover'):
        """Gera sinais de trading baseado em estratégias"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            if data.empty:
                return None
            
            signals = []
            
            if strategy == 'sma_crossover':
                data['SMA_10'] = data['Close'].rolling(10).mean()
                data['SMA_30'] = data['Close'].rolling(30).mean()
                
                for i in range(1, len(data)):
                    if (data['SMA_10'].iloc[i] > data['SMA_30'].iloc[i] and 
                        data['SMA_10'].iloc[i-1] <= data['SMA_30'].iloc[i-1]):
                        signals.append({
                            'date': data.index[i],
                            'signal': 'BUY',
                            'price': data['Close'].iloc[i],
                            'strategy': strategy
                        })
                    elif (data['SMA_10'].iloc[i] < data['SMA_30'].iloc[i] and 
                          data['SMA_10'].iloc[i-1] >= data['SMA_30'].iloc[i-1]):
                        signals.append({
                            'date': data.index[i],
                            'signal': 'SELL',
                            'price': data['Close'].iloc[i],
                            'strategy': strategy
                        })
            
            return signals[-10:] if signals else []  # Últimos 10 sinais
        except Exception as e:
            st.error(f"Erro na geração de sinais: {e}")
            return []
    
    # 8. ANÁLISE DE NOTÍCIAS (Simulado)
    def analyze_news_sentiment(self, symbol):
        """Análise de sentimento de notícias"""
        # Simula notícias (normalmente usaria NewsAPI)
        sample_news = [
            f"{symbol} reaches new highs as investors show confidence",
            f"Market analysts upgrade {symbol} rating to buy",
            f"{symbol} quarterly earnings exceed expectations",
            f"Regulatory concerns may impact {symbol} in short term",
            f"{symbol} announces strategic partnership deal"
        ]
        
        news_analysis = []
        for news in sample_news:
            sentiment = self.analyze_sentiment(news)
            news_analysis.append({
                'headline': news,
                'sentiment_score': sentiment['vader_compound'],
                'sentiment_label': 'Positivo' if sentiment['vader_compound'] > 0.1 else 
                                 'Negativo' if sentiment['vader_compound'] < -0.1 else 'Neutro',
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 24))
            })
        
        return news_analysis
    
    # 9. ANÁLISE COMPORTAMENTAL
    def behavioral_analysis(self, trading_history):
        """Análise de padrões comportamentais do trader"""
        if not trading_history:
            return None
        
        # Simula histórico de trading
        df = pd.DataFrame(trading_history)
        
        analysis = {
            'total_trades': len(df),
            'win_rate': len(df[df['profit'] > 0]) / len(df) * 100 if len(df) > 0 else 0,
            'avg_profit': df['profit'].mean() if len(df) > 0 else 0,
            'avg_holding_time': df['holding_hours'].mean() if 'holding_hours' in df.columns else 0,
            'risk_profile': 'Conservador' if df['profit'].std() < 100 else 'Agressivo',
            'best_day': df.groupby(df['date'].dt.date)['profit'].sum().max() if len(df) > 0 else 0,
            'worst_day': df.groupby(df['date'].dt.date)['profit'].sum().min() if len(df) > 0 else 0
        }
        
        # Sugestões baseadas no comportamento
        suggestions = []
        if analysis['win_rate'] < 50:
            suggestions.append("Considere revisar sua estratégia de entrada")
        if analysis['avg_holding_time'] < 1:
            suggestions.append("Você pode estar fazendo overtrading")
        if analysis['risk_profile'] == 'Agressivo':
            suggestions.append("Considere implementar stop-loss mais rigoroso")
        
        analysis['suggestions'] = suggestions
        return analysis
    
    # 10. CORRELAÇÃO ENTRE ATIVOS
    def calculate_correlations(self, symbols):
        """Calcula matriz de correlação entre ativos"""
        try:
            # Coleta dados
            data = yf.download(symbols, period="1y")['Close']
            
            if data.empty:
                return None, None
            
            # Calcula correlações
            returns = data.pct_change().dropna()
            correlation_matrix = returns.corr()
            
            # Identifica pares com alta correlação para hedging
            hedge_suggestions = []
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        hedge_suggestions.append({
                            'asset1': symbols[i],
                            'asset2': symbols[j],
                            'correlation': corr,
                            'hedge_potential': 'Alto' if abs(corr) > 0.8 else 'Médio'
                        })
            
            return correlation_matrix, hedge_suggestions
        except Exception as e:
            st.error(f"Erro no cálculo de correlações: {e}")
            return None, None