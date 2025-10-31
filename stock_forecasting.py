# stock_forecasting.py

import yfinance as yf
import pandas as pd
from prophet import Prophet
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go

# --------------------------------------------
# External stylesheet
# --------------------------------------------
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&display=swap"
]

# --------------------------------------------
# Fetch historical stock data
# --------------------------------------------
def get_stock_data(ticker):
    try:
        # Download data with auto_adjust=True for accurate close prices
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
        if df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in df.columns
            ]

        df.reset_index(inplace=True)

        # Ensure consistent column names
        if "Date" not in df.columns:
            if "Datetime" in df.columns:
                df.rename(columns={"Datetime": "Date"}, inplace=True)
            else:
                df["Date"] = df.index

        # Convert and clean columns
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Close" not in df.columns:
            # Sometimes Close column may be named differently
            possible_close = [c for c in df.columns if "Close" in c]
            if possible_close:
                df.rename(columns={possible_close[0]: "Close"}, inplace=True)

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# --------------------------------------------
# Forecast using Prophet
# --------------------------------------------
def forecast_stock(df, periods=30):
    try:
        model_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model_df = model_df.dropna()
        model_df["ds"] = pd.to_datetime(model_df["ds"], errors="coerce")
        model_df["y"] = pd.to_numeric(model_df["y"], errors="coerce")
        model_df = model_df.dropna(subset=["ds", "y"])

        # Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(model_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast

    except Exception as e:
        print(f"Prophet error: {e}")
        return None

# --------------------------------------------
# Initialize Dash app
# --------------------------------------------
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Stock Forecast Dashboard"

# --------------------------------------------
# Dropdown options
# --------------------------------------------
stock_options = [
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Google (GOOG)', 'value': 'GOOG'},
    {'label': 'Reliance (RELIANCE.NS)', 'value': 'RELIANCE.NS'},
    {'label': 'TCS (TCS.NS)', 'value': 'TCS.NS'},
    {'label': 'Gold ETF India (GOLDBEES.NS)', 'value': 'GOLDBEES.NS'},
    {'label': 'Gold ETF India (NIPPGOLD.NS)', 'value': 'NIPPGOLD.NS'},
    {'label': 'Gold ETF US (GLD)', 'value': 'GLD'},
    {'label': 'Toyota (TM)', 'value': 'TM'}
]

# --------------------------------------------
# Theme colors
# --------------------------------------------
main_color = "#0057B7"
accent_color = "#FFD700"
background_color = "#F8FAFD"
card_bg = "#FFFFFF"

# --------------------------------------------
# App layout
# --------------------------------------------
app.layout = html.Div([
    html.Div([
        html.Img(
            src="https://img.icons8.com/ios-filled/100/gold-bars.png",
            style={'height': '40px', 'marginRight': '10px'}
        ),
        html.H1(
            "Stock Forecast Dashboard",
            style={
                'fontFamily': 'Inter',
                'fontWeight': '900',
                'margin': '0',
                'fontSize': '2.2rem',
                'color': main_color
            }
        ),
    ], style={
        'textAlign': 'center',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'backgroundColor': background_color,
        'paddingTop': "18px"
    }),

    html.Div([
        html.Div([
            html.Label("Select Asset",
                       style={'fontFamily': 'Inter', 'color': main_color,
                              'fontSize': '1.05rem', 'fontWeight': '600'}),
            dcc.Dropdown(
                id='stock-dropdown',
                options=stock_options,
                value='AAPL',
                clearable=False,
                style={'fontFamily': 'Inter', 'fontWeight': '600', 'backgroundColor': card_bg}
            )
        ], style={'flex': '1', 'minWidth': '240px'}),

        html.Div([
            html.Label("Forecast Period (Days)",
                       style={'fontFamily': 'Inter', 'color': main_color,
                              'fontSize': '1.05rem', 'fontWeight': '600'}),
            dcc.Slider(
                id='forecast-slider',
                min=7, max=90, step=1, value=30,
                marks={7: '7d', 30: '30d', 60: '60d', 90: '90d'}
            )
        ], style={'flex': '2', 'marginLeft': '40px'})
    ], style={
        'display': 'flex',
        'gap': '30px',
        'backgroundColor': card_bg,
        'padding': '32px 32px 24px 32px',
        'borderRadius': '12px',
        'boxShadow': "0 2px 12px rgba(55,60,118,0.05)",
        'width': '74%',
        'margin': '28px auto 0 auto',
        'flexWrap': 'wrap'
    }),

    html.Br(),

    dcc.Tabs(
        id='tabs',
        value='tab1',
        children=[
            dcc.Tab(label='📈 Historical Data', value='tab1'),
            dcc.Tab(label='🔮 Forecast', value='tab2'),
        ],
        style={'backgroundColor': card_bg, 'width': '74%',
               'margin': '0 auto', 'borderRadius': '10px'},
        colors={'border': main_color, 'primary': accent_color, 'background': background_color}
    ),

    html.Div(id='tabs-content', style={'padding': '32px', 'width': '74%', 'margin': '0 auto'})
], style={'backgroundColor': background_color, 'minHeight': '100vh', 'fontFamily': 'Inter'})

# --------------------------------------------
# Callback logic
# --------------------------------------------
@app.callback(
    Output('tabs-content', 'children'),
    Input('stock-dropdown', 'value'),
    Input('forecast-slider', 'value'),
    Input('tabs', 'value')
)
def update_tab(ticker, forecast_days, tab):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        return html.Div("❌ No data found for selected asset or data fetch failed.",
                        style={'color': main_color, 'fontWeight': 'bold'})

    forecast = forecast_stock(df, forecast_days)
    if forecast is None or forecast.empty:
        return html.Div("❌ Forecasting failed. Check data format.",
                        style={'color': main_color, 'fontWeight': 'bold'})

    label = next((x['label'] for x in stock_options if x['value'] == ticker), ticker)

    if tab == 'tab1':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], mode='lines',
            name='Close Price', line=dict(color=main_color)
        ))
        fig.update_layout(
            title=f"{label} - Historical Price",
            xaxis_title='Date', yaxis_title='Price',
            template='plotly_white',
            paper_bgcolor=card_bg, plot_bgcolor=card_bg
        )
        return dcc.Graph(figure=fig)

    elif tab == 'tab2':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], mode='lines',
            name='Actual', line=dict(color=main_color)
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines',
            name='Forecast', line=dict(color=accent_color)
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines', line=dict(width=0), fill='tonexty',
            name='Confidence Interval', fillcolor='rgba(255, 215, 0, 0.12)'
        ))
        fig.update_layout(
            title=f"{label} - {forecast_days}-Day Forecast (Prophet)",
            xaxis_title='Date', yaxis_title='Predicted Price',
            template='plotly_white',
            paper_bgcolor=card_bg, plot_bgcolor=card_bg
        )
        return dcc.Graph(figure=fig)

# --------------------------------------------
# Run the app
# --------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
