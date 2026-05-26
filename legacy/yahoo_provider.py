import yfinance as yf
from data_interface import MarketDataInterface

class yahoo_providetr(MarketDataInterface):
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

    def fetch_data(self, start_date, end_date):
        self.data = yf.download(self.ticker, start=start_date, end=end_date)

    def get_data(self):
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data not fetched yet. Please call fetch_data() first.")
# Example usage:
# yahoo_provider = yahoo_providetr("AAPL")
# yahoo_provider.fetch_data("2020-01-01", "2021-01-
