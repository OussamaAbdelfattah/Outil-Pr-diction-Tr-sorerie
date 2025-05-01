# Installation des librairies requises :
# pip install streamlit prophet plotly pandas scikit-learn numpy statsmodels optuna tensorflow kaleido

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(page_title="📊 Dashboard de Trésorerie", layout="wide")

# Titre de l'application
st.title("📈 Outil Avancé de Prévision de Trésorerie")

# =========================================
# 1. Upload et traitement des données
# =========================================
uploaded_file = st.file_uploader("📂 Charger le fichier Excel de flux de trésorerie", type=["xlsx"])

if uploaded_file:
    # Lecture du fichier Excel
    df_flux = pd.read_excel(uploaded_file, sheet_name="31-12-24")
    
    # Extraction des flux et des dates
    dates = pd.to_datetime(df_flux.columns[1:-1])
    encaissements = df_flux.iloc[0, 1:-1].values.astype(float)
    decaissements = df_flux.iloc[1, 1:-1].values.astype(float)
    
    # Création des DataFrames pour Prophet
    df_enc = pd.DataFrame({'ds': dates, 'y': encaissements})
    df_dec = pd.DataFrame({'ds': dates, 'y': decaissements})
    
    # =========================================
    # 2. Modèles de prévision
    # =========================================
    st.sidebar.header("⚙️ Paramètres des modèles")
    n_mois = st.sidebar.slider("🔢 Nombre de mois à prédire", 1, 24, 6)
    
    # Hyperparamètres Prophet
    st.sidebar.subheader("Paramètres Prophet")
    changepoint_prior_scale = st.sidebar.slider("changepoint_prior_scale", 0.01, 0.5, 0.05)
    seasonality_prior_scale = st.sidebar.slider("seasonality_prior_scale", 0.1, 10.0, 1.0)
    fourier_order = st.sidebar.slider("fourier_order", 3, 15, 5)
    
    # Initialisation des modèles
    model_enc = Prophet(
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model_enc.add_seasonality(name='monthly', period=30.5, fourier_order=fourier_order)
    model_enc.fit(df_enc)
    
    model_dec = Prophet(
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model_dec.add_seasonality(name='monthly', period=30.5, fourier_order=fourier_order)
    model_dec.fit(df_dec)
    
    # Prévisions Prophet
    future_enc = model_enc.make_future_dataframe(periods=n_mois, freq='ME')
    future_dec = model_dec.make_future_dataframe(periods=n_mois, freq='ME')
    
    forecast_enc = model_enc.predict(future_enc)
    forecast_dec = model_dec.predict(future_dec)
    
    # =========================================
    # 5. Comparaison de modèles (ARIMA)
    # =========================================
    st.sidebar.subheader("Paramètres ARIMA")
    arima_p = st.sidebar.slider("ARIMA p", 1, 5, 1)
    arima_d = st.sidebar.slider("ARIMA d", 0, 2, 1)
    arima_q = st.sidebar.slider("ARIMA q", 0, 5, 1)
    
    # Modèle ARIMA pour encaissements
    model_arima_enc = ARIMA(df_enc['y'], order=(arima_p, arima_d, arima_q))
    results_arima_enc = model_arima_enc.fit()
    forecast_arima_enc = results_arima_enc.get_forecast(steps=n_mois)
    
    # Modèle ARIMA pour décaissements
    model_arima_dec = ARIMA(df_dec['y'], order=(arima_p, arima_d, arima_q))
    results_arima_dec = model_arima_dec.fit()
    forecast_arima_dec = results_arima_dec.get_forecast(steps=n_mois)
    
    # Préparation des résultats pour comparaison
    df_comparison = pd.DataFrame({
        'ds': pd.date_range(start=dates[-1], periods=n_mois+1, freq='ME')[1:],
        'yhat_prophet_enc': forecast_enc['yhat'][-n_mois:].values,
        'yhat_arima_enc': forecast_arima_enc.predicted_mean,
        'yhat_prophet_dec': forecast_dec['yhat'][-n_mois:].values,
        'yhat_arima_dec': forecast_arima_dec.predicted_mean
    })
    
    # =========================================
    # 3. KPI Financiers
    # =========================================
    st.header("📊 Dashboard Financier")
    
    # Calcul des KPI
    df_ecart = pd.DataFrame()
    df_ecart['ds'] = forecast_enc['ds']
    df_ecart['encaissement'] = forecast_enc['yhat']
    df_ecart['decaissement'] = forecast_dec['yhat']
    df_ecart['solde'] = df_ecart['encaissement'] - df_ecart['decaissement']
    
    # KPI calculs
    avg_monthly_flow = df_ecart['encaissement'].mean()
    enc_dec_ratio = df_ecart['encaissement'].sum() / df_ecart['decaissement'].sum()
    cumulative_balance = df_ecart['solde'].cumsum().iloc[-1]
    growth_rate = (df_ecart['encaissement'].iloc[-1] - df_ecart['encaissement'].iloc[0]) / df_ecart['encaissement'].iloc[0] * 100
    cash_burn = df_ecart['decaissement'].mean()
    cash_runway = cumulative_balance / cash_burn if cash_burn > 0 else float('inf')
    
    # Affichage des KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Moyenne mensuelle", f"{avg_monthly_flow:,.0f} DH")
    col2.metric("📊 Ratio Enc/Dec", f"{enc_dec_ratio:.2f}")
    col3.metric("🏦 Solde cumulé", f"{cumulative_balance:,.0f} DH")
    col4.metric("📈 Taux croissance", f"{growth_rate:.1f}%")
    
    col5, col6 = st.columns(2)
    col5.metric("🔥 Cash Burn Rate", f"{cash_burn:,.0f} DH/mois")
    col6.metric("⏳ Cash Runway", f"{cash_runway:.1f} mois" if cash_runway != float('inf') else "∞")
    
    # =========================================
    # 4. Visualisations Dashboard
    # =========================================
    tab1, tab2, tab3 = st.tabs(["📈 Flux", "🔍 Comparaison Modèles", "🎮 Simulation"])
    
    with tab1:
        # Graphique waterfall
        fig_waterfall = go.Figure(go.Waterfall(
            name="Trésorerie",
            orientation="v",
            measure=["relative"] * (len(df_ecart)-1) + ["total"],
            x=df_ecart['ds'].dt.strftime('%Y-%m'),
            y=df_ecart['solde'],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Waterfall des Flux de Trésorerie",
            showlegend=True
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Graphiques séparés
        col1, col2 = st.columns(2)
        with col1:
            fig_enc = px.line(df_ecart, x='ds', y='encaissement', title="Encaissements")
            st.plotly_chart(fig_enc, use_container_width=True)
        
        with col2:
            fig_dec = px.line(df_ecart, x='ds', y='decaissement', title="Décaissements")
            st.plotly_chart(fig_dec, use_container_width=True)
        
        # Graphique solde
        fig_solde = px.area(df_ecart, x='ds', y='solde', title="Solde de Trésorerie")
        st.plotly_chart(fig_solde, use_container_width=True)
    
    with tab2:
        # Comparaison des modèles
        st.subheader("Comparaison Prophet vs ARIMA")
        
        fig_comp_enc = go.Figure()
        fig_comp_enc.add_trace(go.Scatter(x=df_comparison['ds'], y=df_comparison['yhat_prophet_enc'], name="Prophet"))
        fig_comp_enc.add_trace(go.Scatter(x=df_comparison['ds'], y=df_comparison['yhat_arima_enc'], name="ARIMA"))
        fig_comp_enc.update_layout(title="Comparaison Encaissements")
        st.plotly_chart(fig_comp_enc, use_container_width=True)
        
        fig_comp_dec = go.Figure()
        fig_comp_dec.add_trace(go.Scatter(x=df_comparison['ds'], y=df_comparison['yhat_prophet_dec'], name="Prophet"))
        fig_comp_dec.add_trace(go.Scatter(x=df_comparison['ds'], y=df_comparison['yhat_arima_dec'], name="ARIMA"))
        fig_comp_dec.update_layout(title="Comparaison Décaissements")
        st.plotly_chart(fig_comp_dec, use_container_width=True)
        
        # Métriques de performance
        mae_prophet_enc = mean_absolute_error(df_enc['y'], forecast_enc['yhat'][:len(df_enc)])
        mae_arima_enc = mean_absolute_error(df_enc['y'], results_arima_enc.fittedvalues)
        
        col1, col2 = st.columns(2)
        col1.metric("MAE Prophet (Enc)", f"{mae_prophet_enc:,.0f} DH")
        col2.metric("MAE ARIMA (Enc)", f"{mae_arima_enc:,.0f} DH")
    
    with tab3:
        # =========================================
        # 8. Simulation de scénarios
        # =========================================
        st.subheader("🎮 Simulation de Scénarios")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            enc_change = st.slider("Variation encaissements (%)", -50, 50, 0)
        with col2:
            dec_change = st.slider("Variation décaissements (%)", -50, 50, 0)
        with col3:
            cash_injection = st.number_input("Injection de trésorerie (DH)", 0, 10000000, 0)
        
        # Application des modifications
        df_simul = df_ecart.copy()
        df_simul['encaissement'] = df_simul['encaissement'] * (1 + enc_change/100)
        df_simul['decaissement'] = df_simul['decaissement'] * (1 + dec_change/100)
        
        if cash_injection > 0:
            df_simul.loc[df_simul['ds'] == df_simul['ds'].min(), 'encaissement'] += cash_injection
        
        df_simul['solde'] = df_simul['encaissement'] - df_simul['decaissement']
        
        # Affichage des résultats
        fig_simul = go.Figure()
        fig_simul.add_trace(go.Bar(x=df_simul['ds'], y=df_simul['encaissement'], name='Encaissements'))
        fig_simul.add_trace(go.Bar(x=df_simul['ds'], y=-df_simul['decaissement'], name='Décaissements'))
        fig_simul.add_trace(go.Scatter(x=df_simul['ds'], y=df_simul['solde'], name='Solde', mode='lines+markers'))
        
        fig_simul.update_layout(
            title="Scénario Simulé",
            barmode='relative',
            hovermode='x unified'
        )
        st.plotly_chart(fig_simul, use_container_width=True)
        
        # Calcul des nouveaux KPI
        new_cumulative = df_simul['solde'].cumsum().iloc[-1]
        new_runway = new_cumulative / (df_simul['decaissement'].mean()) if df_simul['decaissement'].mean() > 0 else float('inf')
        
        st.metric("Nouveau solde cumulé", f"{new_cumulative:,.0f} DH", 
                 delta=f"{(new_cumulative - cumulative_balance):+,.0f} DH")
        st.metric("Nouveau cash runway", 
                 f"{new_runway:.1f} mois" if new_runway != float('inf') else "∞", 
                 delta=f"{(new_runway - cash_runway):+.1f} mois" if cash_runway != float('inf') else "N/A")

    # =========================================
    # 6. Hyperparamétrage automatique
    # =========================================
    if st.sidebar.button("🔍 Optimiser hyperparamètres Prophet"):
        st.sidebar.info("Optimisation en cours...")
        
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'yearly_seasonality': trial.suggest_int('yearly_seasonity', 5, 15),
                'fourier_order': trial.suggest_int('fourier_order', 3, 15)
            }
            
            model = Prophet(**params)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=params['fourier_order'])
            model.fit(df_enc)
            
            df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='60 days')
            pm = performance_metrics(df_cv)
            return np.mean(pm['mape'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        
        st.sidebar.success("Optimisation terminée !")
        st.sidebar.write("Meilleurs paramètres:", study.best_params)
        st.sidebar.write("Meilleur MAPE:", study.best_value)
        
        # Application des meilleurs paramètres
        st.session_state.best_params = study.best_params

# Style CSS personnalisé
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .stMetric label {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stMetric div {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)