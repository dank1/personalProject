import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

url = 'https://api.coinbase.com/v2/prices/BTC-USD/historic?period=week'
response = requests.get(url)

data = response.json()

prices = data['data']['prices']

class account:
    def __init__(self, ammount):
        self.ammount = ammount
        self.price = -1

    def buy(self, startPrice):
        if self.price < 0:
            self.price = startPrice
        else:
            print("you have already bought")

    def sell(self, currentPrice):
        if self.price > 0:
            self.ammount = self.ammount * currentPrice / self.price
            self.price = -1
        else:
            print("you have not bought any")

investmentAccount = account(1000)
index1 = 0
i = 0

# Open a file in read mode
'''
with open('example.txt', 'w') as file:
    # Read the contents of the file
    #content = file.read()
    # Print the content
    #print(content)
    for price in prices:
        timestamp = price['time']
        btc_price = price['price']
        coin_price = float(btc_price)
        if i%2 == 0:
            investmentAccount.buy(coin_price)
        else:
            investmentAccount.sell(coin_price)
        #print(f"ammount={investmentAccount.ammount}")
        dt_object = datetime.fromtimestamp(int(timestamp))
        date_object = dt_object.date()
        #print(f"Date: {dt_object}, BTC Price: {btc_price}")
        file.write(str(dt_object) + ",")
        file.write(btc_price + "\n")
        i +=1
'''

def average(arr):
    return sum(arr) / len(arr)

def geometricAverage(arr):
    sum = np.log10(arr[0])
    for i in range(1, len(arr)):
        sum += np.log10(arr[i])
    return 10 ** (sum / len(arr))

def movingAverage(priceArray, lookback):
    movingAverageArray = [-1.0]
    movingAverageArray.remove(-1.0)
    tempArray = [0.0]
    tempArray.remove(0.0)
    for i2 in range(0, lookback):
        tempArray.append(priceArray[i2])
    for i2 in range(0, lookback - 1):
        movingAverageArray.append(-1.0)
    i3 = 0
    for i2 in range(lookback, len(priceArray)):
        movingAverageArray.append(average(tempArray))
        tempArray.pop(0)
        tempArray.append(priceArray[i2])
        if (i2 == 305):
            i3 += 1
    return movingAverageArray

def geometricMovingAverage(priceArray, lookback):
    movingAverageArray = [-1.0]
    movingAverageArray.remove(-1.0)
    tempArray = [0.0]
    tempArray.remove(0.0)
    for i2 in range(0, lookback):
        tempArray.append(priceArray[i2])
    for i2 in range(0, lookback - 1):
        movingAverageArray.append(-1.0)
    for i2 in range(lookback, len(priceArray)):
        movingAverageArray.append(geometricAverage(tempArray))
        tempArray.pop(0)
        tempArray.append(priceArray[i2])
    return movingAverageArray

def sd(priceArray, lookback):
    movingAverageArray = [-1.0]
    movingAverageArray.remove(-1.0)
    tempArray = [0.0]
    tempArray.remove(0.0)
    for i2 in range(0, lookback):
        tempArray.append(priceArray[i2])
    for i2 in range(0, lookback - 1):
        movingAverageArray.append(-1.0)
    for i2 in range(lookback, len(priceArray)):
        movingAverageArray.append(float(np.std(tempArray)))
        tempArray.pop(0)
        tempArray.append(priceArray[i2])
    return movingAverageArray

def bollinger_bands(priceArray, lookback, k=2):
    """
    Calculates Bollinger Bands.

    Args:
        priceArray: A list of prices.
        lookback: The lookback period for the moving average and standard deviation.
        k: The number of standard deviations for the upper and lower bands.

    Returns:
        A tuple containing three lists: upper_band, middle_band, lower_band.
    """
    middle_band = movingAverage(priceArray, lookback)
    std_dev = sd(priceArray, lookback)

    upper_band = []
    lower_band = []

    for i in range(len(middle_band)):
        if middle_band[i] == -1.0:
            upper_band.append(-1.0)
            lower_band.append(-1.0)
        else:
            upper_band.append(middle_band[i] + (std_dev[i] * k))
            lower_band.append(middle_band[i] - (std_dev[i] * k))
    return upper_band, middle_band, lower_band

def rsi(priceArray, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    Args:
        priceArray: A list of prices.
        period: The period for calculating RSI (default is 14).
    Returns:
        A list containing RSI values. First (period) values will be -1.0 (not enough data).
    """
    if len(priceArray) < period + 1:
        return [-1.0] * len(priceArray)
    # Calculate price changes
    price_changes = []
    for i in range(1, len(priceArray)):
        price_changes.append(priceArray[i] - priceArray[i - 1])
    # Initialize RSI array with -1.0 for first period values
    rsi_values = [-1.0] * period
    # Calculate gains and losses
    gains = []
    losses = []

    for change in price_changes:
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    # Calculate initial average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate RSI for the first valid point
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    # Calculate RSI for remaining points using exponential smoothing
    for i in range(period + 1, len(priceArray)):
        # Update average gain and loss using exponential smoothing
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        # Calculate RSI
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi_values

def trading_strategy(prices, short_ma=5, long_ma=15, rsi_period=14, initial_balance=1000):
    """
    Simple trading strategy:
    - Buy when short MA crosses above long MA and RSI < 30
    - Sell when short MA crosses below long MA or RSI > 70
    """
    ma_short = movingAverage(prices, short_ma)
    ma_long = movingAverage(prices, long_ma)
    rsi_vals = rsi(prices, rsi_period)

    position = 0  # 0 = not holding, 1 = holding
    buy_price = 0
    balance = initial_balance
    trade_log = []

    # Find the starting index where all indicators have valid values
    start_idx = max(short_ma, long_ma, rsi_period)

    # Calculate the offset for moving average arrays
    ma_offset = short_ma - 1  # This is how many -1.0 values are at the beginning

    for i in range(start_idx, len(prices)):
        # Calculate the corresponding index in the moving average arrays
        ma_idx = i - ma_offset

        # Check if we have valid indicator values and indices are within bounds
        if (ma_idx >= 0 and ma_idx < len(ma_short) and ma_idx < len(ma_long) and
                ma_short[ma_idx] != -1.0 and ma_long[ma_idx] != -1.0 and rsi_vals[i] != -1.0 and
                ma_idx > 0 and ma_short[ma_idx - 1] != -1.0 and ma_long[ma_idx - 1] != -1.0):

            # Buy condition
            if (ma_short[ma_idx] > ma_long[ma_idx] and ma_short[ma_idx - 1] <= ma_long[ma_idx - 1] and rsi_vals[
                i] < 30 and position == 0):
                position = 1
                buy_price = prices[i]
                trade_log.append((i, 'BUY', prices[i]))
            # Sell condition
            elif (position == 1 and (
                    (ma_short[ma_idx] < ma_long[ma_idx] and ma_short[ma_idx - 1] >= ma_long[ma_idx - 1]) or rsi_vals[
                i] > 70)):
                position = 0
                balance = balance * (prices[i] / buy_price)
                trade_log.append((i, 'SELL', prices[i]))

    # If still holding at the end, sell at last price
    if position == 1:
        balance = balance * (prices[-1] / buy_price)
        trade_log.append((len(prices) - 1, 'SELL', prices[-1]))
    return balance, trade_log


def plot_trading_results(prices, trades, ma_short=None, ma_long=None, rsi_vals=None):
    plt.figure(figsize=(14, 8))
    plt.plot(prices, label='Price', color='black')

    # Plot moving averages if provided
    if ma_short is not None:
        plt.plot(ma_short, label='Short MA', color='blue', alpha=0.6)
    if ma_long is not None:
        plt.plot(ma_long, label='Long MA', color='red', alpha=0.6)

    # Plot buy/sell signals
    buy_signals = [idx for idx, action, price in trades if action == 'BUY']
    buy_prices = [price for idx, action, price in trades if action == 'BUY']
    sell_signals = [idx for idx, action, price in trades if action == 'SELL']
    sell_prices = [price for idx, action, price in trades if action == 'SELL']
    plt.scatter(buy_signals, buy_prices, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_signals, sell_prices, marker='v', color='red', label='Sell', s=100)

    plt.title('Trading Strategy Results')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally plot RSI
    '''
    if rsi_vals is not None:
        plt.figure(figsize=(14, 3))
        plt.plot(rsi_vals, label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(30, color='green', linestyle='--', alpha=0.5)
        plt.title('RSI')
        plt.xlabel('Index')
        plt.ylabel('RSI Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        '''

print("Moving-Average-1  Moving-Average-2")

pricesFloat = [0.0]
with open('example.txt', 'r') as file:
    i1 = 0
    content = " "
    dateFileString = ""
    price = 0.00
    i4 = 0
    while (content != ""):
        content = file.readline(20)
        if (i1 % 2 == 0):
            date = content[0:19]
        else:
            coin_price = float(content)
            pricesFloat.insert(0, coin_price)
        i1 += 1
        if (i1 == 304):
            i4 += 1
    # prices.remove(0.0)
    # sdArray = sd(prices, 3)
    pricesFloat.remove(0.0)
    lookback1 = 2
    lookback2 = 3
    maArray1 = movingAverage(pricesFloat, lookback1)
    maArray2 = movingAverage(pricesFloat, lookback2)

    # Calculate RSI with default period of 14
    rsi_values = rsi(pricesFloat, period=14)

    # Example RSI trading strategy
    print("RSI Values (first 10):")
    for i in range(min(10, len(rsi_values))):
        if rsi_values[i] != -1.0:
            print(f"RSI[{i}]: {rsi_values[i]:.2f}")

    # Example: RSI overbought/oversold signals
    print("\nRSI Trading Signals:")
    for i in range(len(rsi_values)):
        if rsi_values[i] != -1.0:
            if rsi_values[i] > 70:
                print(f"Overbought signal at index {i}: RSI = {rsi_values[i]:.2f}")
            elif rsi_values[i] < 30:
                print(f"Oversold signal at index {i}: RSI = {rsi_values[i]:.2f}")

    # Run trading strategy
    final_balance, trades = trading_strategy(pricesFloat, short_ma=5, long_ma=15, rsi_period=14, initial_balance=1000)
    print(f"\nFinal account value after strategy: ${final_balance:.2f}")
    print("Trade log:")
    for idx, action, price in trades:
        print(f"{action} at index {idx}, price: {price}")

    # Visualize results
    ma_short = movingAverage(pricesFloat, 5)
    ma_long = movingAverage(pricesFloat, 15)
    rsi_vals = rsi(pricesFloat, 14)
    plot_trading_results(pricesFloat, trades, ma_short=ma_short, ma_long=ma_long, rsi_vals=rsi_vals)

    isBought = False
    maxk = 2
    maxj = 3
    maxValue = 1000.0
    for i2 in range(0, len(maArray1)):
        print(str(maArray1[i2]) + ", " + str(maArray2[i2]))
    '''
    for j in range(3, 21):
        lookback2 = j
        for k in range(2, lookback2):
            lookback1 = k
            maArray1 = movingAverage(pricesFloat, lookback1)
            maArray2 = movingAverage(pricesFloat, lookback2)
            for i2 in range(lookback2, len(prices) - 1):
                # print(f"{maArray1[i2]}  {maArray2[i2]}")
                if (maArray1[i2] > maArray2[i2] and maArray1[i2 - 1] < maArray2[i2 - 1] and not isBought):
                    investmentAccount.buy(prices[i2])
                    isBought = True
                elif (maArray1[i2] < maArray2[i2] and maArray1[i2 - 1] > maArray2[i2 - 1] and isBought):
                    investmentAccount.sell(prices[i2])
                    isBought = False
            if (isBought):
                investmentAccount.sell(prices[len(prices) - 1])
                isBought = False
            print(str(k) + ", " + str(j) + ", " + str(investmentAccount.ammount))
            ''''''
            if(investmentAccount.ammount > maxValue):
                maxValue = investmentAccount.ammount
                maxk = k
                maxj = j
                print(str(maxk) + ", " + str(maxj) + ", " + str(maxValue))
                ''''''
            investmentAccount = account(1000)'''