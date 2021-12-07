import datetime
import streamlit as st
from PIL import Image
import pandas as pd
import pandas_datareader as web
from sklearn.svm import SVR

title       = st.container()
header      = st.container()
dataset     = st.container()
time        = st.container()
end_page    = st.container()

with title:
    st.title("Welcome to the stock market website")
    image_ = Image.open('image_stock.jpg')
    st.image(image_, use_column_width=True, width= 10000)

with header:
    st.header("""This website will practice simple analysis and visualize data from stock's data. In addition, we will the predict stock's price in a short-time.""")
    st.markdown("🔐 To analyze the company stock you want:")
    st.markdown("🗝️ Choose **_START DATE_** and **_END DATE_**")
    st.markdown("🗝️ Choose **_the company's code you want_** and there are **_four fields_** with four codes for each:")
    st.markdown("+ **_Financial_**")
    st.markdown("+ **_Technology Service_**")
    st.markdown("+ **_Electronic Technology_**")
    st.markdown("+ **_Retail_**")
    st.markdown("**_Note that: the base code on website was chosen is BKR-A. Please, choose another code._**")
    st.sidebar.header("Input information")

with time:
    start_time                  = st.sidebar.date_input("Start date", datetime.date(2021, 12, 1))
    end_time                    = st.sidebar.date_input("End date")

with dataset:
    kind_of_stock               = ["Financial", "Technology_Service", "Electronic_Technology", "Retail"]
    Financial                   = ["BRK-A", "JPM", "V", "BAC"]
    Technology_Service          = ["MSFT", "GOOGL", "FB", "ADBE"]
    Electronic_Technology       = ["AAPL", "NVDA", "TSM", "QCOM"]
    Retail                      = ["AMZN", "HD", "WMT", "BABA"]
    stock_code                  = st.sidebar.selectbox("Company's stocks in United States", kind_of_stock)
    if stock_code              == kind_of_stock[0]:
        choice                  = st.sidebar.selectbox(kind_of_stock[0], Financial)
    elif stock_code            == kind_of_stock[1]:
        choice                  = st.sidebar.selectbox(kind_of_stock[1], Technology_Service)
    elif stock_code            == kind_of_stock[2]:
        choice                  = st.sidebar.selectbox(kind_of_stock[2], Electronic_Technology)
    elif stock_code            == kind_of_stock[3]:
        choice                  = st.sidebar.selectbox(kind_of_stock[3], Retail)

def get_data_web(symbol):
    """Get data of the company's code from DataReader"""
    if symbol  == choice:
        df      = web.DataReader(choice, "yahoo", start= start_time, end=end_time)
        return df

def get_data_csv(symbol):
    """Get data of the company's code from the files.csv"""
    if symbol  == choice:
        df      = pd.read_csv(f"{choice}.csv")
        return df

def draw_chart():
    """Draw the charts about Total volume, Close, High, Low prices"""
    Close           = get_data_web(choice)["Close"]
    High            = get_data_web(choice)["High"]
    Low             = get_data_web(choice)["Low"]
    three_price     = pd.concat([Close, High, Low], axis=1)
    Total_Volume    =  get_data_web(choice)["Close"] * get_data_web(choice)["Volume"]
    returns         = (get_data_web(choice)["Close"] / get_data_web(choice)["Close"].shift(1)) - 1
    st.area_chart(three_price)
    st.write(f"🌈**_Total volume of {choice}_**")
    st.line_chart(Total_Volume)
    st.write(f"🌈**_The return for each day of {choice}_**")
    st.area_chart(returns)
    st.caption("✏️**_The points you see on the graph that below the x-axis is negative returns_**")

def predict_stock():
    """Predict stock price of the next 'n' years"""
    #create an empty list to store the independent
    df                      = get_data_csv(choice)
    last_day                = df["Adj Close"].tail(1)
    df                      = df.head(len(df)-1)
    days                    = list()
    adj_close_prices        = list()
    df_days                 = df.loc[:, "Date"]
    df_adj_close_prices     = df.loc[:, "Adj Close"]
    for day in df_days:
        days.append([int(day.split("-")[2])])
    for adj_close_price in df_adj_close_prices:
        adj_close_prices.append(float(adj_close_price))

    #Create and train a SVR model using a linear kernel
    lin_svr = SVR(kernel="linear", C = 1000.0)
    lin_svr.fit(days, adj_close_prices)

    #Creat and train a SVR model using a polynomial
    poly_svr = SVR(kernel="poly", C = 1000.0, degree=2)
    poly_svr.fit(days, adj_close_prices)

    #Create and train a SVR model using a rbf kernel
    rbf_svr = SVR(kernel= 'rbf', C=1000.0, gamma=0.15)
    rbf_svr.fit(days, adj_close_prices)

    #charts
    df1         = pd.DataFrame(poly_svr.predict(days), columns=["Poly"])
    df2         = pd.DataFrame(rbf_svr.predict(days), columns=["RBF"])
    df3         = pd.DataFrame(lin_svr.predict(days), columns=["Lin"])
    da_frame    = pd.concat([df1, df2, df3], axis=1)
    st.line_chart(da_frame)

    day         = [[int((st.number_input("Input the number of day you want to predict:")))]]
    st.write("Price that is predicted by RBF: ",rbf_svr.predict(day))
    st.write("Price that is predicted by Lin: ",lin_svr.predict(day))
    st.write("Price that is predicted by Poly:",poly_svr.predict(day))
    st.write("Price of stock in the last day:", last_day)

if __name__ == "__main__":
    with dataset:
        dataset.write(f"🌈 **_The table below shows the stock's data of {choice}_**")
        dataset.write(get_data_web(choice))
        dataset.write(f"🌈 **_The chart below shows the Close, Low, and High price of {choice}_**")
        draw_chart()
        dataset.header("Predict stock's price")
        dataset.write("In this part, we use the **_files.csv_** which is download from the **_finance.yahoo website_** in a year ago.")
        dataset.markdown("🚀 **_Visualization_**")
        dataset.markdown("🚀 **_Give stock's price_** which is predicted by the number of days")
        predict_stock()
    with end_page:
        st.title("------------------------------------------------------------------------------------------------")
        individual_image = Image.open("hong.jpg")
        st.header("🚀 About us")
        st.image(individual_image, width=400)
        st.write("**_Creator_**: Phạm Văn Hồng")
        st.write("**_Email_**: vanhong542002@gmail.com")
        st.write("**_Phone number_**: 0812768***")
        st.write("**_Working as sophomore at National Economic University._**")
        st.write("**_If you need any additional assistance, please contact me by email above._**")


