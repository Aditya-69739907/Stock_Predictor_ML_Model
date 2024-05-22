
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as me
import xgboost as xgb

START="2020-06-20"
END="2024-05-30"
ticker=input("Enter STOCK TICKER:")
df=yf.download(ticker,start=START, end=END)


a=df.reset_index()


df1=a.drop(["Date","Adj Close"],axis=1)


ma50=df1.Close.rolling(50).mean()


plt.figure(figsize=(12, 6))
plt.plot(df1.Close, label='Original Price')
plt.plot(ma50, 'r', label='Predicted Price')
plt.legend()
plt.title("Price vs 50MA")

ma100=df1.Close.rolling(100).mean()


plt.figure(figsize=(16,8))
plt.plot(df1.Close,label="Actual Price")
plt.plot(ma50,'g',label="50MA")
plt.plot(ma100,'r',label="100MA")
plt.legend()
plt.title("Actual Price vs 50MA vs 100MA")

"""OBSERVATION**:Where the 50MA(Green),100MA(Red) line intersects  the uptrend of the price starts."""



#Testing and Training the model


x=df1[["Close"]]
y=df1["Close"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = xgb.XGBRegressor(Objective='reg:squarederror',learning_rate=0.1) 
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#Model Evaluation

mse=me(y_test,y_pred)
print("mean_squared_error:",mse)

#Plotting actual vs Predicted Graph
import plotly.graph_objs as go
import plotly.io as pio

# Create a figure
fig = go.Figure()

# Add the original price line
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values,
                         mode='lines', name='Original Price'))

# Add the predicted price line
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_pred,
                         mode='lines', name='Predicted Price',
                         line=dict(color='red')))

# Update the layout
fig.update_layout(
    title='Original vs Predicted Prices',
    xaxis_title='Index',
    yaxis_title='Price',
    autosize=False,
    width=1800,
    height=900,
    legend=dict(x=0, y=1)
)

# Add zoom functionality
fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

# Show the plot
pio.show(fig)



future_date = input("Enter a future date (YYYY-MM-DD): ")


future_date = pd.to_datetime(future_date)

# Predict the price for the future date
future_price = model.predict([[df1.loc[df1.index[-1], 'Close']]])[0]

# Print the predicted price
print("The predicted price for", future_date.strftime('%Y-%m-%d'), "is:", future_price)
