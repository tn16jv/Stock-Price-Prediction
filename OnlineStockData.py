import pandas_datareader.data as pdr


def get_stock_data(ticker=None, start_date=None, end_date=None):
    valid = False
    if ticker is None:
        ticker = input("Enter a stock ticker: ")
    while not valid:
        try:
            data = pdr.get_data_yahoo(ticker, start_date, end_date)
            valid = True
        except:
            print("Invalid, try again.")
            ticker = input("Enter a stock ticker: ")
    return data.reset_index(), ticker
