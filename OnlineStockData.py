import pandas_datareader.data as pdr


def get_stock_data(start, end):
    valid = False
    while not valid:
        try:
            ticker = input("Enter a stock ticker: ")
            data = pdr.get_data_yahoo(ticker, start, end)
            valid = True
        except:
            print("Invalid, try again.")
    return data
