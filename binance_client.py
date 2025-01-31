import pandas as pd
from binance import Client
import config
import matplotlib.pyplot as plt

client = Client(config.API_KEY, config.SECRET_KEY)


def main():
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
    klines = pd.DataFrame(klines)
    klines = klines.drop(columns=klines.columns[6:])
    klines.columns = ["dt", "open", "high", "low", "close", "volume"]
    klines["dt"] = pd.to_datetime(klines["dt"], unit="ms")
    for column in klines.columns[1:]:
        klines[column] = klines[column].astype(float)
    print(klines)
    print(klines.shape)
    #klines.to_csv(rf"Database\1h_{len(klines)}n_btc.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(klines["dt"], klines["close"])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    
    
if __name__ == "__main__":
    main()
