import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from stocknews import StockNews

st.title('Stock Price Prediction')
st.sidebar.header('BlamerX')

ticker_symbol = st.sidebar.text_input('Enter a Ticker Symbol', 'ETH-USD')

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start)).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end)).strftime("%Y-%m-%d")
no_of_news = st.sidebar.slider("Number of News You Want", 1, 15, 3)

st.sidebar.warning("Disclaimer: This site is for educational purposes only. Do not use the information provided for real trading or investment decisions.")

# Retrieve the full name for the ticker symbol
ticker = yf.Ticker(ticker_symbol)
ticker_info = ticker.info
full_name = ticker_info.get('longName', '')
description = ticker_info.get('description','')
longbusinesssummary=ticker_info.get('longBusinessSummary','')
st.subheader(f'{full_name} - {ticker_symbol}')
st.write(description)
st.write(longbusinesssummary)

# Download stock price data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

fig = px.line(data, x=data.index, y=data['Adj Close'])
fig.update_layout(title={
    'text': f'{ticker_symbol} - {full_name}',
    'x': 0.5})
st.plotly_chart(fig)

pricing_data,news,predictions=st.tabs(["Pricing Data",f"Top {no_of_news} News", "Predictions"])

with pricing_data:
    st.header("Price Movements")
    data2 = data
    data2['% Change'] = data["Adj Close"] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2["% Change"].mean() * 252 * 100
    if annual_return > 0:
        st.write("Annual Return is ", ":arrow_up:", annual_return, '%')
    else:
        st.write("Annual Return is ", ":arrow_down:", annual_return, '%')
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.write('Standard Deviation is ', stdev * 100, '%')
    st.write('Risk Adj. Return is ', annual_return / (stdev * 100))

with news:
    st.header(f'News of {ticker_symbol} - {full_name}')
    sn = StockNews(ticker_symbol, save_news=False)
    df_news = sn.read_rss()
    for i in range(no_of_news):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

with predictions:
    st.write("Comming Soon")
