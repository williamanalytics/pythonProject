import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import time
from PIL import Image
from prophet.plot import add_changepoints_to_plot


image = Image.open('BB.png')
sel_col, disp_col, col_3 = st.columns((1, 2, 1 ))
disp_col.image(image)
st.markdown("What is the **''possible''** futures price of the 18 months ahead such as: commodities, stocks, REITS, ETFs...")
st.markdown("See below some examples just copy & paste the tickers or take a look into the link.")
st.markdown(" **Examples**: MSFT, GOOG, GME, GC=F(Gold), CL=F(Oil), BRFS3.SA, PETR3.SA, ETH-USD...")
st.write("[https://finance.yahoo.com/lookup](https://finance.yahoo.com/lookup)")

#BODY

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
sel_col, dis_col = st.columns(2)
input_feature = sel_col.text_input('Which ticker would you like to type ? Ex: Gold', 'GC=F')
n_month = st.slider("Months of predcitions:", 1, 18)
period = n_month * 30


@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data = load_data(input_feature)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress.progress(i + 1)

#FORECASTING

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#TRAINING THE CODE

m = Prophet(weekly_seasonality=True,yearly_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period, include_history=True)
forecast = m.predict(future)
st.subheader("Forecast data")
fig = plot_plotly(m, forecast, xlabel='Date', ylabel='Price')
st.plotly_chart(fig,  use_container_width=True)
st.subheader("Forecast Components")

with st.expander("Explanation"):
            st.markdown("""**Trend**: Shows the information about the where the trend suppose to go.""")
            st.write("**Weekly / Yearly**: Shows the asset performance trend in a period of time.")

fig2 = m.plot_components(forecast)
st.write(fig2)

#CHANGE POINTS
st.subheader('Change Points Plot')
st.markdown(' The change points are points that represents rapid changes movements inside the trajectory.')
st.markdown(' **Attention**: by default Prophet takes into account only the first 80% of the history.')
fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
st.write(fig3)

#DISCLOSURE
st.subheader("The Bottom Line")
st.write("""
    It is a little a bit utopian to be able to see into the future, but the technical analysis offers us the most important concepts in forecasting based on historical 
    data over price of an asset.""")

st.write(""" In this project was used Prophet model by Facebook Prophet (Machine Learning).
    It is an artificial intelligence based on the idea that systems can learn from data, basically using Fourier series.""")

st.markdown("""However, the market is driven by expectations, some announcements affect and move with the sentiment market, investors who have 
    losses from previous decisions are more likely to be affected by good and bad news alike.
    The rational expectations theory suggests that expectations and outcome are linked and the decisions are influenced by all 
    available informations and experiences from previous mistakes and success.
    The Adaptive expectations theory suggest that people who expect price to rise will continue to do so the next period. """)

st.subheader("Disclamer:")

st.markdown(" **This is not financial tool for personal investment decision is only for educational purpose.** ")

