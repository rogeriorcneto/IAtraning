# Interface principal para a Trading Suite
from trading_suite_app import TradingSuite
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def main():
    # Inicializa a Trading Suite
    suite = TradingSuite()
    
    # Header principal
    st.markdown('<h1 class="main-header">🚀 Trading Suite - 10 Aplicativos Integrados</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("📊 Navegação")
    app_choice = st.sidebar.selectbox(
        "Escolha o Aplicativo:",
        [
            "🏠 Dashboard Principal",
            "📊 1. Análise de Sentimento", 
            "🔮 2. Análise Preditiva",
            "⚡ 3. Trading HFT",
            "💰 4. Arbitragem", 
            "📈 5. Otimização de Portfólio",
            "🛡️ 6. Gerenciamento de Risco",
            "🤖 7. Bot de Trading",
            "📰 8. Análise de Notícias",
            "🧠 9. Análise Comportamental",
            "🔗 10. Correlação de Ativos"
        ]
    )
    
    # Configurações globais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configurações")
    
    # Símbolos para análise
    default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    symbols_input = st.sidebar.text_area(
        "Símbolos (separados por vírgula):",
        value=", ".join(default_symbols)
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Valor do investimento
    investment_amount = st.sidebar.number_input(
        "Valor do Investimento ($):",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Dashboard Principal
    if app_choice == "🏠 Dashboard Principal":
        render_dashboard(suite, symbols, investment_amount)
    
    # App 1: Análise de Sentimento
    elif app_choice == "📊 1. Análise de Sentimento":
        render_sentiment_analysis(suite, symbols)
    
    # App 2: Análise Preditiva
    elif app_choice == "🔮 2. Análise Preditiva":
        render_predictive_analysis(suite, symbols)
    
    # App 3: Trading HFT
    elif app_choice == "⚡ 3. Trading HFT":
        render_hft_trading(suite, symbols)
    
    # App 4: Arbitragem
    elif app_choice == "💰 4. Arbitragem":
        render_arbitrage(suite, symbols)
    
    # App 5: Otimização de Portfólio
    elif app_choice == "📈 5. Otimização de Portfólio":
        render_portfolio_optimization(suite, symbols, investment_amount)
    
    # App 6: Gerenciamento de Risco
    elif app_choice == "🛡️ 6. Gerenciamento de Risco":
        render_risk_management(suite, symbols, investment_amount)
    
    # App 7: Bot de Trading
    elif app_choice == "🤖 7. Bot de Trading":
        render_trading_bot(suite, symbols)
    
    # App 8: Análise de Notícias
    elif app_choice == "📰 8. Análise de Notícias":
        render_news_analysis(suite, symbols)
    
    # App 9: Análise Comportamental
    elif app_choice == "🧠 9. Análise Comportamental":
        render_behavioral_analysis(suite)
    
    # App 10: Correlação de Ativos
    elif app_choice == "🔗 10. Correlação de Ativos":
        render_correlation_analysis(suite, symbols)

def render_dashboard(suite, symbols, investment_amount):
    """Dashboard principal com resumo de todos os aplicativos"""
    st.header("📊 Dashboard Principal")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ativos Monitorados", len(symbols))
    with col2:
        st.metric("Valor do Portfólio", f"${investment_amount:,}")
    with col3:
        # Simula performance
        performance = np.random.uniform(-5, 15)
        st.metric("Performance (%)", f"{performance:.2f}%", f"{performance:.2f}%")
    with col4:
        st.metric("Alertas Ativos", np.random.randint(0, 10))
    
    st.markdown("---")
    
    # Resumo rápido dos principais aplicativos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Previsões Rápidas")
        for symbol in symbols[:3]:
            predictions, _ = suite.predict_prices(symbol, days=7)
            if predictions:
                change = (predictions[-1] - predictions[0]) / predictions[0] * 100
                st.write(f"**{symbol}**: {change:+.2f}% (7 dias)")
    
    with col2:
        st.subheader("📊 Sentimento de Mercado")
        for symbol in symbols[:3]:
            sentiment, _ = suite.get_market_sentiment(symbol)
            if sentiment:
                score = sentiment['compound']
                label = "😊 Positivo" if score > 0.1 else "😟 Negativo" if score < -0.1 else "😐 Neutro"
                st.write(f"**{symbol}**: {label} ({score:.2f})")

def render_sentiment_analysis(suite, symbols):
    """Interface para análise de sentimento"""
    st.header("📊 Análise de Sentimento")
    
    symbol = st.selectbox("Escolha o ativo:", symbols)
    
    if st.button("Analisar Sentimento"):
        with st.spinner("Analisando sentimento..."):
            sentiment, sentiments = suite.get_market_sentiment(symbol)
            
            if sentiment:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score Composto", f"{sentiment['compound']:.3f}")
                with col2:
                    st.metric("Positivo", f"{sentiment['positive']:.3f}")
                with col3:
                    st.metric("Negativo", f"{sentiment['negative']:.3f}")
                
                # Gráfico de sentimento
                fig = go.Figure(data=[
                    go.Bar(name='Positivo', x=[symbol], y=[sentiment['positive']]),
                    go.Bar(name='Negativo', x=[symbol], y=[sentiment['negative']])
                ])
                fig.update_layout(title="Análise de Sentimento")
                st.plotly_chart(fig, use_container_width=True)

def render_predictive_analysis(suite, symbols):
    """Interface para análise preditiva"""
    st.header("🔮 Análise Preditiva")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Escolha o ativo:", symbols)
    with col2:
        days = st.slider("Dias para previsão:", 1, 60, 30)
    
    if st.button("Gerar Previsão"):
        with st.spinner("Gerando previsão..."):
            predictions, data = suite.predict_prices(symbol, days)
            
            if predictions and data is not None:
                # Gráfico de previsão
                fig = go.Figure()
                
                # Dados históricos
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Histórico',
                    line=dict(color='blue')
                ))
                
                # Previsões
                future_dates = pd.date_range(
                    start=data.index[-1] + timedelta(days=1),
                    periods=days,
                    freq='D'
                )
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines',
                    name='Previsão',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Previsão de Preços - {symbol}",
                    xaxis_title="Data",
                    yaxis_title="Preço ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas da previsão
                current_price = data['Close'].iloc[-1]
                predicted_price = predictions[-1]
                change = (predicted_price - current_price) / current_price * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Preço Atual", f"${current_price:.2f}")
                with col2:
                    st.metric("Previsão", f"${predicted_price:.2f}")
                with col3:
                    st.metric("Mudança Esperada", f"{change:+.2f}%")

def render_hft_trading(suite, symbols):
    """Interface para trading HFT"""
    st.header("⚡ Trading de Alta Frequência")
    
    if st.button("Buscar Oportunidades HFT"):
        with st.spinner("Escaneando oportunidades..."):
            opportunities = suite.hft_scanner(symbols)
            
            if opportunities:
                df = pd.DataFrame(opportunities)
                
                # Tabela de oportunidades
                st.subheader("🎯 Oportunidades Detectadas")
                for _, opp in df.iterrows():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{opp['symbol']}**")
                    with col2:
                        color = "green" if opp['signal'] == 'BUY' else "red"
                        st.markdown(f"<span style='color:{color}'>{opp['signal']}</span>", 
                                  unsafe_allow_html=True)
                    with col3:
                        st.write(f"${opp['price']:.2f}")
                    with col4:
                        st.write(f"Força: {opp['strength']:.1f}")
            else:
                st.info("Nenhuma oportunidade HFT detectada no momento.")

def render_arbitrage(suite, symbols):
    """Interface para arbitragem"""
    st.header("💰 Análise de Arbitragem")
    
    symbol = st.selectbox("Ativo para arbitragem:", ["BTC-USD", "ETH-USD", "ADA-USD"])
    
    if st.button("Buscar Oportunidades de Arbitragem"):
        with st.spinner("Buscando oportunidades..."):
            arbitrage = suite.find_arbitrage_opportunities(symbol)
            
            if arbitrage:
                st.success("🎯 Oportunidade de Arbitragem Encontrada!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Comprar em:** {arbitrage['buy_exchange']}")
                    st.write(f"**Preço:** ${arbitrage['buy_price']:.2f}")
                with col2:
                    st.write(f"**Vender em:** {arbitrage['sell_exchange']}")
                    st.write(f"**Preço:** ${arbitrage['sell_price']:.2f}")
                
                st.metric("Lucro Potencial", f"{arbitrage['profit_pct']:.3f}%")
            else:
                st.info("Nenhuma oportunidade de arbitragem detectada.")

def render_portfolio_optimization(suite, symbols, investment_amount):
    """Interface para otimização de portfólio"""
    st.header("📈 Otimização de Portfólio")
    
    if st.button("Otimizar Portfólio"):
        with st.spinner("Otimizando portfólio..."):
            optimization = suite.optimize_portfolio(symbols, investment_amount)
            
            if optimization:
                st.success("✅ Portfólio Otimizado!")
                
                # Métricas do portfólio
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Esperado", f"{optimization['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Volatilidade", f"{optimization['volatility']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{optimization['sharpe_ratio']:.3f}")
                
                # Alocação recomendada
                st.subheader("💼 Alocação Recomendada")
                allocations = optimization['allocations']
                
                weights = [allocations[symbol]['weight'] for symbol in symbols]
                amounts = [allocations[symbol]['amount'] for symbol in symbols]
                
                # Gráfico de pizza
                fig = px.pie(values=weights, names=symbols, title="Distribuição do Portfólio")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela detalhada
                allocation_df = pd.DataFrame([
                    {
                        'Ativo': symbol,
                        'Peso (%)': f"{allocations[symbol]['weight']*100:.1f}%",
                        'Valor ($)': f"${allocations[symbol]['amount']:.2f}"
                    }
                    for symbol in symbols
                ])
                st.dataframe(allocation_df, use_container_width=True)

def render_risk_management(suite, symbols, investment_amount):
    """Interface para gerenciamento de risco"""
    st.header("🛡️ Gerenciamento de Risco")
    
    symbol = st.selectbox("Ativo para análise:", symbols)
    position_size = st.number_input("Tamanho da posição ($):", value=investment_amount//len(symbols))
    
    if st.button("Analisar Risco"):
        with st.spinner("Analisando riscos..."):
            risk_metrics = suite.risk_analysis(symbol, position_size)
            
            if risk_metrics:
                st.subheader("📊 Métricas de Risco")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR 95%", f"${risk_metrics['var_95_amount']:.2f}")
                    st.metric("VaR 99%", f"${risk_metrics['var_99_amount']:.2f}")
                with col2:
                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown_pct']:.2f}%")
                    st.metric("Volatilidade Anual", f"{risk_metrics['volatility_pct']:.2f}%")
                with col3:
                    st.metric("Perda Máxima Estimada", f"${risk_metrics['max_loss_estimate']:.2f}")
                
                # Alertas de risco
                st.subheader("⚠️ Alertas de Risco")
                if risk_metrics['var_95_amount'] > position_size * 0.1:
                    st.warning("Alto risco detectado: VaR 95% > 10% da posição")
                if risk_metrics['volatility_pct'] > 50:
                    st.warning("Alta volatilidade detectada")
                if abs(risk_metrics['max_drawdown_pct']) > 30:
                    st.error("Drawdown máximo muito alto!")

def render_trading_bot(suite, symbols):
    """Interface para bot de trading"""
    st.header("🤖 Bot de Trading Automatizado")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Ativo:", symbols)
    with col2:
        strategy = st.selectbox("Estratégia:", ["sma_crossover", "rsi_oversold", "breakout"])
    
    if st.button("Gerar Sinais"):
        with st.spinner("Gerando sinais de trading..."):
            signals = suite.generate_trading_signals(symbol, strategy)
            
            if signals:
                st.subheader("📊 Sinais Recentes")
                
                signals_df = pd.DataFrame(signals)
                signals_df['date'] = pd.to_datetime(signals_df['date']).dt.strftime('%Y-%m-%d')
                
                for _, signal in signals_df.iterrows():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(signal['date'])
                    with col2:
                        color = "green" if signal['signal'] == 'BUY' else "red"
                        st.markdown(f"<span style='color:{color}'><b>{signal['signal']}</b></span>", 
                                  unsafe_allow_html=True)
                    with col3:
                        st.write(f"${signal['price']:.2f}")
                    with col4:
                        st.write(signal['strategy'])
            else:
                st.info("Nenhum sinal gerado para o período selecionado.")

def render_news_analysis(suite, symbols):
    """Interface para análise de notícias"""
    st.header("📰 Análise de Notícias")
    
    symbol = st.selectbox("Ativo:", symbols)
    
    if st.button("Analisar Notícias"):
        with st.spinner("Analisando notícias..."):
            news_analysis = suite.analyze_news_sentiment(symbol)
            
            if news_analysis:
                st.subheader("📊 Sentimento das Notícias")
                
                for news in news_analysis:
                    with st.expander(f"{news['sentiment_label']} - {news['timestamp'].strftime('%H:%M')}"):
                        st.write(news['headline'])
                        st.write(f"**Score:** {news['sentiment_score']:.3f}")
                
                # Resumo do sentimento
                avg_sentiment = np.mean([n['sentiment_score'] for n in news_analysis])
                positive_count = len([n for n in news_analysis if n['sentiment_score'] > 0.1])
                negative_count = len([n for n in news_analysis if n['sentiment_score'] < -0.1])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentimento Médio", f"{avg_sentiment:.3f}")
                with col2:
                    st.metric("Notícias Positivas", positive_count)
                with col3:
                    st.metric("Notícias Negativas", negative_count)

def render_behavioral_analysis(suite):
    """Interface para análise comportamental"""
    st.header("🧠 Análise Comportamental do Trader")
    
    # Simula histórico de trading
    st.info("💡 Esta é uma demonstração com dados simulados. Conecte seu histórico real para análise completa.")
    
    if st.button("Analisar Comportamento"):
        # Gera dados simulados de trading
        trading_history = []
        for i in range(100):
            trading_history.append({
                'date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'profit': np.random.normal(10, 50),
                'holding_hours': np.random.exponential(24)
            })
        
        analysis = suite.behavioral_analysis(trading_history)
        
        if analysis:
            st.subheader("📊 Estatísticas de Trading")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Trades", analysis['total_trades'])
            with col2:
                st.metric("Taxa de Acerto", f"{analysis['win_rate']:.1f}%")
            with col3:
                st.metric("Lucro Médio", f"${analysis['avg_profit']:.2f}")
            with col4:
                st.metric("Perfil de Risco", analysis['risk_profile'])
            
            # Sugestões
            if analysis['suggestions']:
                st.subheader("💡 Sugestões de Melhoria")
                for suggestion in analysis['suggestions']:
                    st.write(f"• {suggestion}")

def render_correlation_analysis(suite, symbols):
    """Interface para análise de correlação"""
    st.header("🔗 Correlação entre Ativos")
    
    if st.button("Calcular Correlações"):
        with st.spinner("Calculando correlações..."):
            correlation_matrix, hedge_suggestions = suite.calculate_correlations(symbols)
            
            if correlation_matrix is not None:
                # Heatmap de correlações
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(x="Ativo", y="Ativo", color="Correlação"),
                    x=symbols,
                    y=symbols,
                    color_continuous_scale='RdBu_r',
                    title="Matriz de Correlação"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sugestões de hedging
                if hedge_suggestions:
                    st.subheader("💡 Sugestões de Hedging")
                    
                    hedge_df = pd.DataFrame(hedge_suggestions)
                    st.dataframe(hedge_df, use_container_width=True)
                else:
                    st.info("Nenhuma oportunidade de hedging identificada com os critérios atuais.")

if __name__ == "__main__":
    main()
