import pandas as pd
from model_manager import Model


def main():
    df = pd.read_csv(r"Database//1m_28801n_btc.csv")
    df = df.iloc[len(df) - 12000:len(df)-8500]
    model_manager = Model()
    #model_manager.create("1m_trading_td3", "TradingEnv", df, 32, ["open", "high", "low", "close", "volume"], [512, 512], [400, 300])
    #model_manager.learn("1m_trading_td3", df, 9000, 25)
    model_manager.test("1m_trading_td3_col7", df)


if __name__ == '__main__':
    main()
