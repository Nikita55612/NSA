import pandas as pd
from datetime import timedelta
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
import matplotlib.pyplot as plt
from model_manager import Model
import config


def compound(units, nano):
    if units >= 0 and nano >= 0:
        return float(f"{units}.{nano:09d}")
    else:
        return -float(f"{abs(units)}.{abs(nano):09d}")


def main():
    with Client(config.TOKEN) as client:
        candles = []
        life_candle = None
        for candle in client.get_all_candles(
            figi="BBG006L8G4H1",
            from_=now() - timedelta(days=44),
            interval=CandleInterval.CANDLE_INTERVAL_DAY
        ):
            if candle.is_complete:
                candles.append([candle.time,
                                compound(candle.open.units, candle.open.nano),
                                compound(candle.high.units, candle.high.nano),
                                compound(candle.low.units, candle.low.nano),
                                compound(candle.close.units, candle.close.nano),
                                candle.volume])
            else:
                life_candle = [candle.time,
                               compound(candle.open.units, candle.open.nano),
                               compound(candle.high.units, candle.high.nano),
                               compound(candle.low.units, candle.low.nano),
                               compound(candle.close.units, candle.close.nano),
                               candle.volume]

        candles = pd.DataFrame(candles, columns=["dt", "open", "high", "low", "close", "volume"])
        #candles.to_csv(f"Database//1h_{len(candles)}n_yndx.csv")
        action, next_action = Model().predict("1d_trading_td3_yndx", candles)
        c_price = candles.iloc[-1]['close']
        if life_candle:
            candles.loc[len(candles.index)] = life_candle
            action_, next_action_ = Model().predict("1d_trading_td3_yndx", candles)
            l_price = candles.iloc[-1]['close']
        else:
            action_, next_action_ = 0, 0
        t_action = ["null", "buy", "sell"]
        s_action = ["<>", ">", "<"]
        print("CANDLE_INTERVAL_DAY")
        print(f"{t_action[action]} ({s_action[action]} {c_price}) =>\n"
              f"{t_action[next_action]} ({s_action[next_action]} {l_price}) =>\n"
              f"{t_action[next_action_]} ({s_action[next_action_]} x)\n")
        candles = []
        life_candle = None
        for candle in client.get_all_candles(
            figi="BBG006L8G4H1",
            from_=now() - timedelta(days=4),
            interval=CandleInterval.CANDLE_INTERVAL_HOUR
        ):
            if candle.is_complete:
                candles.append([candle.time,
                                compound(candle.open.units, candle.open.nano),
                                compound(candle.high.units, candle.high.nano),
                                compound(candle.low.units, candle.low.nano),
                                compound(candle.close.units, candle.close.nano),
                                candle.volume])
            else:
                life_candle = [candle.time,
                               compound(candle.open.units, candle.open.nano),
                               compound(candle.high.units, candle.high.nano),
                               compound(candle.low.units, candle.low.nano),
                               compound(candle.close.units, candle.close.nano),
                               candle.volume]

        candles = pd.DataFrame(candles, columns=["dt", "open", "high", "low", "close", "volume"])
        #candles.to_csv(f"Database//1h_{len(candles)}n_yndx.csv")
        action, next_action = Model().predict("1h_trading_td3_yndx1", candles)
        c_price = candles.iloc[-1]['close']
        if life_candle:
            candles.loc[len(candles.index)] = life_candle
            action_, next_action_ = Model().predict("1h_trading_td3_yndx1", candles)
            l_price = candles.iloc[-1]['close']
        else:
            action_, next_action_ = 0, 0
        t_action = ["null", "buy", "sell"]
        s_action = ["<>", ">", "<"]
        print("CANDLE_INTERVAL_HOUR")
        print(f"{t_action[action]} ({s_action[action]} {c_price}) =>\n"
              f"{t_action[next_action]} ({s_action[next_action]} {l_price}) =>\n"
              f"{t_action[next_action_]} ({s_action[next_action_]} x)\n")
        candles = []
        life_candle = None
        for candle in client.get_all_candles(
            figi="BBG006L8G4H1",
            from_=now() - timedelta(days=200),
            interval=CandleInterval.CANDLE_INTERVAL_15_MIN
        ):
            if candle.is_complete:
                candles.append([candle.time,
                                compound(candle.open.units, candle.open.nano),
                                compound(candle.high.units, candle.high.nano),
                                compound(candle.low.units, candle.low.nano),
                                compound(candle.close.units, candle.close.nano),
                                candle.volume])
            else:
                life_candle = [candle.time,
                               compound(candle.open.units, candle.open.nano),
                               compound(candle.high.units, candle.high.nano),
                               compound(candle.low.units, candle.low.nano),
                               compound(candle.close.units, candle.close.nano),
                               candle.volume]

        candles = pd.DataFrame(candles, columns=["dt", "open", "high", "low", "close", "volume"])
        #candles.to_csv(f"Database//1h_{len(candles)}n_yndx.csv")
        Model().test("15m_trading_td3_yndx1", candles.iloc[:4200])
        action, next_action = Model().predict("15m_trading_td3_yndx1", candles)
        c_price = candles.iloc[-1]['close']
        if life_candle:
            candles.loc[len(candles.index)] = life_candle
            action_, next_action_ = Model().predict("15m_trading_td3_yndx1", candles)
            l_price = candles.iloc[-1]['close']
        else:
            action_, next_action_ = 0, 0
        t_action = ["null", "buy", "sell"]
        s_action = ["<>", ">", "<"]
        print("CANDLE_INTERVAL_15_MIN")
        print(f"{t_action[action]} ({s_action[action]} {c_price}) =>\n"
              f"{t_action[next_action]} ({s_action[next_action]} {l_price}) =>\n"
              f"{t_action[next_action_]} ({s_action[next_action_]} x)\n")
        candles = []
        life_candle = None
        for candle in client.get_all_candles(
            figi="BBG006L8G4H1",
            from_=now() - timedelta(days=400),
            interval=CandleInterval.CANDLE_INTERVAL_30_MIN
        ):
            if candle.is_complete:
                candles.append([candle.time,
                                compound(candle.open.units, candle.open.nano),
                                compound(candle.high.units, candle.high.nano),
                                compound(candle.low.units, candle.low.nano),
                                compound(candle.close.units, candle.close.nano),
                                candle.volume])
            else:
                life_candle = [candle.time,
                               compound(candle.open.units, candle.open.nano),
                               compound(candle.high.units, candle.high.nano),
                               compound(candle.low.units, candle.low.nano),
                               compound(candle.close.units, candle.close.nano),
                               candle.volume]

        candles = pd.DataFrame(candles, columns=["dt", "open", "high", "low", "close", "volume"])
        #candles.to_csv(f"Database//30m_{len(candles)}n_yndx.csv")
        input()
        action, next_action = Model().predict("1d_trading_td3_yndx", candles)
        c_price = candles.iloc[-1]['close']
        if life_candle:
            candles.loc[len(candles.index)] = life_candle
            action_, next_action_ = Model().predict("1d_trading_td3_yndx", candles)
            l_price = candles.iloc[-1]['close']
        else:
            action_, next_action_ = 0, 0
        t_action = ["null", "buy", "sell"]
        s_action = ["<>", ">", "<"]
        print("CANDLE_INTERVAL_DAY")
        print(f"{t_action[action]} ({s_action[action]} {c_price}) =>\n"
              f"{t_action[next_action]} ({s_action[next_action]} {l_price}) =>\n"
              f"{t_action[next_action_]} ({s_action[next_action_]} x)\n")
    return 0


if __name__ == "__main__":
    main()
