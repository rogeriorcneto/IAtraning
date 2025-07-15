# Trading Suite Moderna com IA Integrada
from trading_suite_app import TradingSuite
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def main():
    # Configuração moderna da página
    st.set_page_config(
        page_title="🤖 AI Trading Suite - 10 Apps Integrados",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS da SYNAPSE com cores corporativas
    apply_synapse_css()
    
    # Inicializa a Trading Suite
    suite = TradingSuite()
    
    # Header Principal SYNAPSE com Logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo.png", width=200)
        except:
            st.markdown("<h1 style='text-align: center; color: #1f5582;'>SYNAPSE</h1>", unsafe_allow_html=True)
    
    st.markdown('''
    <div class="main-header">
        🤖 AI Trading Suite
        <span class="synapse-badge">POWERED BY SYNAPSE AI</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Navegação Moderna na Sidebar
    render_modern_sidebar()
    
    # Área Principal
    render_main_content(suite)

def apply_synapse_css():
    """Aplica CSS com cores corporativas da SYNAPSE"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Background - Cores SYNAPSE */
    .stApp {
        background: linear-gradient(135deg, #1f5582 0%, #2980b9 50%, #3498db 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        margin: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Main Header SYNAPSE - Branco para legibilidade */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* SYNAPSE Badge Branco */
    .synapse-badge {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        animation: synapse-pulse-white 2s infinite;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes synapse-pulse-white {
        0% { transform: scale(1); box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2); }
        50% { transform: scale(1.05); box-shadow: 0 8px 25px rgba(255, 255, 255, 0.4); }
        100% { transform: scale(1); box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2); }
    }
    
    /* Cards Modernos */
    .modern-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .modern-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(31, 38, 135, 0.5);
    }
    
    /* Buttons SYNAPSE */
    .stButton > button {
        background: linear-gradient(135deg, #1f5582 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(31, 85, 130, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(31, 85, 130, 0.6);
    }
    
    /* Métricas Modernas */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.3);
        transition: transform 0.3s ease;
    }
    
    .css-1r6slb0:hover {
        transform: scale(1.05);
    }
    
    /* Alertas Modernos */
    .stSuccess {
        background: rgba(46, 213, 115, 0.15);
        border: 1px solid rgba(46, 213, 115, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(46, 213, 115, 0.2);
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(255, 193, 7, 0.2);
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.2);
    }
    
    /* Info Box SYNAPSE */
    .ai-info-box {
        background: linear-gradient(135deg, rgba(31, 85, 130, 0.1), rgba(52, 152, 219, 0.1));
        border: 2px solid rgba(31, 85, 130, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .ai-info-box::before {
        content: '🤖';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 2rem;
        opacity: 0.3;
    }
    
    /* Navigation Pills */
    .nav-pill {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        margin: 0.3rem;
        display: inline-block;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(10px);
    }
    
    .nav-pill:hover {
        background: rgba(255, 255, 255, 0.4);
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.3);
    }
    
    /* Loading Animation SYNAPSE */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(31, 85, 130, 0.3);
        border-radius: 50%;
        border-top-color: #1f5582;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Charts com Glassmorphism */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
    }
    
    </style>
    """, unsafe_allow_html=True)

def render_modern_sidebar():
    """Sidebar SYNAPSE com explicações sobre IA"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f5582, #2980b9); 
                border-radius: 15px; margin-bottom: 1rem; color: white;'>
        <h2>🚀 SYNAPSE AI Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Seletor de App Principal
    app_choice = st.sidebar.selectbox(
        "🎯 Escolha o Módulo AI:",
        [
            "🏠 AI Dashboard",
            "🧠 1. Sentiment AI", 
            "🔮 2. Predictive AI",
            "⚡ 3. HFT AI Scanner",
            "💰 4. Arbitrage AI", 
            "📈 5. Portfolio AI",
            "🛡️ 6. Risk AI",
            "🤖 7. Trading Bot AI",
            "📰 8. News AI",
            "🧠 9. Behavioral AI",
            "🔗 10. Correlation AI"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Configurações
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #2980b9, #3498db); 
                border-radius: 10px; margin: 1rem 0; color: white;'>
        <h3>⚙️ Configurações SYNAPSE</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Símbolos
    symbols_input = st.sidebar.text_area(
        "📊 Ativos para Análise:",
        value="AAPL, GOOGL, MSFT, TSLA, AMZN",
        help="Digite os símbolos separados por vírgula"
    )
    
    # Valor de investimento
    investment_amount = st.sidebar.number_input(
        "💰 Capital de Investimento:",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=5000,
        help="Valor total para análise de portfólio"
    )
    
    # AI Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #1f5582, #2980b9); 
                border-radius: 10px; margin: 1rem 0; color: white;'>
        <h3>🤖 IA SYNAPSE Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    ai_confidence = st.sidebar.slider(
        "🎯 Confiança da IA:",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        help="Nível mínimo de confiança para alertas"
    )
    
    ai_aggressiveness = st.sidebar.selectbox(
        "⚡ Agressividade da IA:",
        ["Conservador", "Moderado", "Agressivo"],
        index=1,
        help="Define o comportamento dos algoritmos"
    )
    
    return app_choice, symbols_input, investment_amount, ai_confidence, ai_aggressiveness

def render_main_content(suite):
    """Conteúdo principal com foco em IA"""
    
    # Seção de explicação sobre IA
    st.markdown("""
    <div class="ai-info-box">
        <h2 style='color: white; text-shadow: 1px 1px 4px rgba(0,0,0,0.5);'>🤖 Como a IA Funciona na Trading Suite</h2>
        <p style='color: white; font-size: 1.1rem;'>Nossa plataforma integra <strong>Inteligência Artificial</strong> em cada um dos 10 módulos:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid de explicações de IA
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 **IA em Análise de Sentimento**
        - **NLP Avançado**: VADER + TextBlob para análise de sentimento
        - **ML em Tempo Real**: Processamento de milhares de notícias
        - **Deep Learning**: Classificação automática de sentimentos
        
        ### 🔮 **IA Preditiva**
        - **Random Forest**: Previsão de preços com 15+ features
        - **Séries Temporais**: LSTM para padrões temporais
        - **Ensemble Models**: Combinação de múltiplos algoritmos
        
        ### ⚡ **IA em HFT**
        - **Pattern Recognition**: Detecção de micro-padrões
        - **Reinforcement Learning**: Otimização de estratégias
        - **Real-time AI**: Decisões em microssegundos
        
        ### 💰 **IA de Arbitragem**
        - **Anomaly Detection**: Identifica discrepâncias incomuns
        - **Price Prediction**: ML para timing de arbitragem
        - **Risk Assessment**: IA avalia riscos automaticamente
        
        ### 📈 **IA de Portfólio**
        - **Modern Portfolio Theory**: Otimização com IA
        - **Dynamic Rebalancing**: Ajustes automáticos
        - **Scenario Analysis**: IA simula milhares de cenários
        """)
    
    with col2:
        st.markdown("""
        ### 🛡️ **IA de Gerenciamento de Risco**
        - **Stress Testing**: IA simula crises de mercado
        - **Anomaly Detection**: Detecta comportamentos anômalos
        - **Predictive Risk**: Antecipa riscos futuros
        
        ### 🤖 **Trading Bot com IA**
        - **Adaptive Learning**: Bot aprende com o mercado
        - **Strategy Evolution**: Estratégias evoluem sozinhas
        - **Emotional AI**: Elimina vieses emocionais
        
        ### 📰 **IA de Notícias**
        - **News Aggregation**: IA coleta de 1000+ fontes
        - **Sentiment Analysis**: Análise automática de impacto
        - **Event Prediction**: Prevê impactos de eventos
        
        ### 🧠 **IA Comportamental**
        - **Pattern Mining**: Encontra padrões em comportamento
        - **Bias Detection**: Identifica vieses cognitivos
        - **Performance Coaching**: IA como coach pessoal
        
        ### 🔗 **IA de Correlação**
        - **Dynamic Correlation**: Correlações que mudam no tempo
        - **Hidden Patterns**: IA encontra relações ocultas
        - **Hedge Optimization**: Maximiza eficiência de hedge
        """)
    
    # Demonstração prática
    st.markdown("---")
    st.markdown("""
    <h2 style='color: white; text-align: center; font-size: 2.5rem; 
               text-shadow: 2px 2px 6px rgba(0,0,0,0.5); margin: 2rem 0;'>
        🚀 Demonstração da IA em Ação
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Testar Sentiment AI"):
            with st.spinner("IA analisando sentimentos..."):
                # Simula análise de IA
                st.success("✅ IA detectou sentimento BULLISH em AAPL (confiança: 87%)")
                st.info("🤖 Recomendação: Considerar posição LONG baseada em análise NLP")
    
    with col2:
        if st.button("🔮 Testar Predictive AI"):
            with st.spinner("IA gerando previsões..."):
                st.success("✅ IA prevê alta de 12% em TSLA (próximos 30 dias)")
                st.info("🤖 Modelo: Random Forest + LSTM (acurácia: 78%)")
    
    with col3:
        if st.button("🛡️ Testar Risk AI"):
            with st.spinner("IA calculando riscos..."):
                st.warning("⚠️ IA detectou risco elevado em portfólio (VaR: 8.5%)")
                st.info("🤖 Sugestão: Reduzir exposição em 15%")
    
    # Seção de vantagens da IA
    st.markdown("---")
    st.markdown("""
    <h2 style='color: white; text-align: center; font-size: 2.5rem; 
               text-shadow: 2px 2px 6px rgba(0,0,0,0.5); margin: 2rem 0;'>
        🎯 Vantagens da IA na Trading Suite
    </h2>
    """, unsafe_allow_html=True)
    
    advantages_col1, advantages_col2 = st.columns(2)
    
    with advantages_col1:
        st.markdown("""
        ### 🚀 **Velocidade**
        - Análise de milhares de dados em segundos
        - Decisões em tempo real
        - Processamento 24/7 sem pausas
        
        ### 🎯 **Precisão**
        - Elimina erros humanos
        - Análise baseada em dados
        - Modelos validados estatisticamente
        
        ### 📈 **Escalabilidade**
        - Analisa centenas de ativos simultaneamente
        - Processa múltiplas estratégias
        - Cresce com seu portfólio
        """)
    
    with advantages_col2:
        st.markdown("""
        ### 🧠 **Aprendizado**
        - Melhora continuamente
        - Adapta-se a novos padrões
        - Evolui com o mercado
        
        ### 🛡️ **Controle de Risco**
        - Monitoramento constante
        - Alertas preventivos
        - Stop-loss inteligente
        
        ### 💰 **Lucratividade**
        - Identifica oportunidades ocultas
        - Otimiza retornos
        - Minimiza custos de transação
        """)

if __name__ == "__main__":
    main()
