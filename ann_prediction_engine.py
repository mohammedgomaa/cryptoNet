from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import json as json
from datetime import datetime, timedelta
import pandas as pd
from keras.utils import np_utils
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from keras.preprocessing import sequence

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

def groupData(logs, timeDelta):
        earliestTimestampLog = min(logs, key=lambda x:x['timestamp'])
        largestTimestampLog = max(logs, key=lambda x:x['timestamp'])

        earliestTimestamp = earliestTimestampLog['timestamp']

        totalTimeDuration = largestTimestampLog['timestamp'] - earliestTimestampLog['timestamp']

        totalTimeDurationSeconds = totalTimeDuration.total_seconds()

        numberTicks = int(round(totalTimeDurationSeconds/timeDelta))

        groupedData = []
        timeDeltaDuration = timedelta(seconds=timeDelta)

        for num in range(0,numberTicks):
            startDeltaDuration = timedelta(seconds=(num*timeDelta))
            startTime = earliestTimestamp + startDeltaDuration
            endTime = startTime + timeDeltaDuration - timedelta(seconds=1)
            period = {'index': num, 'startTime': startTime, 'endTime': endTime, 'data': []}
            groupedData.append(period)

        for log in logs:
            for period in groupedData:
                if log['timestamp'] > period['startTime'] and log['timestamp'] < period['endTime']:
                    period['data'].append(log)

        return groupedData


def calculateBidValue(initialOrderBook, bids):

        bidStatistics = {}

        for rateLevel in initialOrderBook['orderBook']['bids']:
            rate = rateLevel[0]
            amount = rateLevel[1]
            bidStatistics[rate] = { 'lastAmount': amount, 'aggregateAmountChange': 0 }

        for bid in bids:
            rate = bid['rate']
            if rate in bidStatistics:
                rateLevel = bidStatistics[rate]
                amountChange = bid['amount'] - rateLevel['lastAmount'] 
                rateLevel['aggregateAmountChange'] = rateLevel['aggregateAmountChange'] + amountChange
                rateLevel['lastAmount'] = bid['amount']
            else:
                bidStatistics[rate] = { 'lastAmount': bid['amount'], 'aggregateAmountChange': bid['amount'] }

        totalBidValue = 0
        for bid in bidStatistics:
            totalBidValue = totalBidValue + bidStatistics[bid]['aggregateAmountChange']

        return totalBidValue




def calculateAskValue(initialOrderBook, asks):

        askStatistics = {}

        for rateLevel in initialOrderBook['orderBook']['asks']:
            rate = rateLevel[0]
            amount = rateLevel[1]
            askStatistics[rate] = { 'lastAmount': amount, 'aggregateAmountChange': 0 }

        for ask in asks:
            rate = ask['rate']
            if rate in askStatistics:
                rateLevel = askStatistics[rate]
                amountChange = ask['amount'] - rateLevel['lastAmount'] 
                rateLevel['aggregateAmountChange'] = rateLevel['aggregateAmountChange'] + amountChange
                rateLevel['lastAmount'] = ask['amount']
            else:
                askStatistics[rate] = { 'lastAmount': ask['amount'], 'aggregateAmountChange': ask['amount'] }

        totalAskValue = 0
        for ask in askStatistics:
            totalAskValue = totalAskValue + askStatistics[ask]['aggregateAmountChange']

        return totalAskValue

  

  

def generatePeriodMetaData(thisTickData, nextTickData):

        orderBooks = filter(lambda x: x['event'] == 'newOrderBook', thisTickData)
        initialOrderBook = min(orderBooks, key=lambda x:x['timestamp'])

        trades = filter(lambda x: x['event'] == 'newTrade', thisTickData)
        nextTickTrades = filter(lambda x: x['event'] == 'newTrade', nextTickData)
        tradeRequests = filter(lambda x: x['event'] == 'orderBookModify', thisTickData)
        bids = filter(lambda x: x['type'] == 'bid', tradeRequests)
        asks = filter(lambda x: x['type'] == 'ask', tradeRequests)
        tickers = filter(lambda x: x['event'] == 'ticker', thisTickData)

        # need to calculate totalTradeValue (sum of rate time qty on all trades)
        
        totalBidValue = calculateBidValue(initialOrderBook, bids)
        totalAskValue = calculateAskValue(initialOrderBook, asks)

        firstLevelBidValue = calculateFirstLevelBidValue(initialOrderBook, bids)
        firstLevelAskValue = calculateFirstLevelAskValue(initialOrderBook, asks)



        orderBookCount = len(orderBooks)
        tradeCount = len(trades)
        bidCount = len(bids)
        askCount = len(asks)
        tickerCount = len(tickers)


        # other features to generate...
        # 1. Feature 30: Volume of the first level of bid
        # 2. Feature 10: Volume of the first level of ask
        # 3. Feature 99: Mean volume of the first ten levels of bid
        # 4. Feature 112: Derivative of the tenth level of ask price
        # 5. Feature 101: Accumulated difference of volumes
        # 6. Feature 132: Derivative of the tenth level of bid price
        # 7. Feature 12: Volume of the third level of ask
        # 8. Feature 97: Mean volume of the first ten levels of ask
        # 9. Feature 11: Volume of the second level of ask
        # 10. Feature 13: Volume of the fourth level of ask

        print ''
        print 'totalBidValue: ', totalBidValue
        print 'totalAskValue: ', totalAskValue
        print 'tradeCount: ', tradeCount
        print 'bidCount: ', bidCount
        print 'askCount: ', askCount

        

        highestTradePriceThisTick = max(trades, key=lambda x: float(x['rate']))
        highestTradePriceNextTick = max(nextTickTrades, key=lambda x: float(x['rate']))

        highestTradePriceThisTick = float(highestTradePriceThisTick['rate'])
        highestTradePriceNextTick = float(highestTradePriceNextTick['rate'])
        print 'highestTradePriceThisTick', highestTradePriceThisTick
        print 'highestTradePriceNextTick', highestTradePriceNextTick
        
        pricePercentDifference = ((highestTradePriceNextTick - highestTradePriceThisTick)/highestTradePriceThisTick)*100

        # need to impliment logic to calculate future profit by comparisons of both highest and lowest trade price

        if pricePercentDifference > .2:
            action = 'buy'
        elif pricePercentDifference < -.2:
            action = 'sell'
        else:
             action = 'hold'

        print 'pricePercentDifference: ', pricePercentDifference
        print 'action: ', action

        metadata = {}
        metadata['totalBidValue'] = totalBidValue
        metadata['totalAskValue'] = totalAskValue
        metadata['tradeCount'] = tradeCount
        metadata['bidCount'] = bidCount
        metadata['askCount'] = askCount
        metadata['action'] = action


        return metadata

def reformatLogs(logs):
    reformattedLogs = []
    for log in logs:
        jsonLog = json.loads(log)
        jsonLog['timestamp'] = datetime.strptime(jsonLog['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')

        if jsonLog['event'] in ['orderBookRemove']:
            jsonLog['rate'] = float(jsonLog['rate'])

        if jsonLog['event'] in ['orderBookModify', 'newTrade']:
            jsonLog['amount'] = float(jsonLog['amount'])
            # jsonLog['rate'] = float(jsonLog['rate'])
            jsonLog['total'] = float(jsonLog['total'])

        if jsonLog['event'] in ['ticker'] :
            jsonLog['lowestAsk'] = float(jsonLog['lowestAsk'])
            jsonLog['highestBid'] = float(jsonLog['highestBid'])
            jsonLog['percentChange'] = float(jsonLog['percentChange'])
            jsonLog['baseVolume'] = float(jsonLog['baseVolume'])
            jsonLog['quoteVolume'] = float(jsonLog['quoteVolume'])
            jsonLog['24hourHigh'] = float(jsonLog['24hourHigh'])
            jsonLog['24hourLow'] = float(jsonLog['24hourLow'])
    
        reformattedLogs.append(jsonLog)
    return reformattedLogs


def generateMetaData(reformattedLogs, timeDelta):
    groupedData = groupData(reformattedLogs, timeDelta)

    for index in range(0, len(groupedData)-1):
        metadata = generatePeriodMetaData(groupedData[index]['data'], groupedData[index+1]['data'])
        groupedData[index]['metadata'] = metadata      

    for period in groupedData:
        del period['data']

    metaData = groupedData


    metaData = map(lambda x: x['metadata'], metaData[:-1])

    return metaData

def main():
    timeDelta = 60*5
    print 'running crypoNet'

    print 'extracting logs from file'
    logs = open('./data/eth_btc.json')

    print 'reformatting logs'
    logs = reformatLogs(logs)
    print len(logs)
    
    print 'generating metadata from logs'
    metaData = generateMetaData(logs, timeDelta)

    #create dataframe 
    metaDataDF = pd.DataFrame(metaData)

    # split dataframe into x and y 
    x = metaDataDF.iloc[:, 1:6].values
    y = metaDataDF.iloc[:, 0].values

    # one hot encode labels
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)

    # scale x values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # split data into training, validation, and test sets
    trainingSizePercent = .60
    validateSizePercent = .40

    trainingSize = int(len(x)*trainingSizePercent)
    validateSize = int(trainingSize*validateSizePercent)

    xTrain = x[:trainingSize]
    yTrain = y[:trainingSize]

    xTest = x[trainingSize:]
    yTest = y[trainingSize:]

    xTrain = xTrain[validateSize:]
    xValidate = xTrain[:validateSize]

    yTrain = yTrain[validateSize:]
    yValidate = yTrain[:validateSize]

    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    print len(xTrain)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(xTrain, yTrain, epochs=20, batch_size=len(xTrain)/5, validation_data=(xValidate, yValidate))

    yPredict = model.predict(xTest)
    yPredict = np.round(yPredict, 2)
    print yPredict
    # yPredict = (yPredict > 0.6)

    print yTest
    print yPredict 
    # print (yTest==yPredict).all()

    results = model.evaluate(xTest, yTest)
    print results

    # print 
    # 
    # yTest = (yTest > 0.5)
    # print yPredict
    # print yTest

    # accuracy = accuracy_score(yTest, yPredict)
    # print(accuracy)

    # yPredict = (yPredict > 0.5)
    # yTest = (yTest > .5)
    # print yPredict
    # print yTest

    # model.save_weights('./model.hdf5')

    # with open('./model.json', 'w') as f:
    #     f.write(model.to_json())

main()