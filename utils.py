'''
#about 4% of the data are missed
#might need to do collaboration filtering first for data preprocessing
#after that normalize and do RNN
'''
import numpy as np
import csv
import sklearn.preprocessing as prep
import pandas as pd
import matplotlib.pyplot as plt
#load data

def data_read():
	raw_data = pd.read_csv("./data/2017April2018Nov.csv")
	#unix time change
	raw_data['Timestamp'] = pd.to_datetime(raw_data['Timestamp'],unit='s')
	#data preprocessing
	proc_data = raw_data.dropna()
	
	print("shape of raw_data: ", raw_data.shape)
	print("num missing vals: ", raw_data['Open'].isna().sum())
	print("-------------")
	print(proc_data.shape)
	print(proc_data['Open'].isna().sum())

	proc_data[['Volume_(BTC)','Weighted_Price']] = proc_data[['Weighted_Price', 'Volume_(BTC)']]
	return proc_data

def minmax_scaler(in_data):
    num_samples, n_x, n_y = in_data.shape
    final_data = []
    for i in range(num_samples):
        cur_data = in_data[i]
        preprocessor = prep.MinMaxScaler(feature_range=(0, 10)).fit(cur_data)
        proc_data = preprocessor.transform(cur_data)
        final_data.append(proc_data)
    return final_data

def preprocessing_data(stock, seq_len):
    amount_features = len(stock.columns)
    data = stock.values
    data = data[:][:,2:]
    
    sequence_length = seq_len + 1
    temp_test = []
    for index in range(len(data) - sequence_length):
        temp_test.append(data[index : index + sequence_length])
        
    result = np.array(temp_test) # samples, seq_len, num_features
    result = np.asarray(minmax_scaler(result))
    #train, result = standard_scaler(train, result)
    row = round(0.95 * result.shape[0])
    train = result[:int(row)]
    
    
    X_train = train[:, : -1]
    y_train = train[:, -1][: ,-1]
    X_test = result[int(row) :, : -1]
    y_test = result[int(row) :, -1][ : ,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_features))  
    y_train = y_train.reshape((y_train.shape[0], 1))
    return [X_train.astype(np.float), y_train.astype(np.float), X_test.astype(np.float), y_test.astype(np.float)]


#plt.plot(proc_data.Timestamp, proc_data.Open, 'r--', 
#	proc_data.Timestamp, proc_data['Volume_(BTC)'], 'g-')
#plt.show()



# raw_data.to_csv("out.csv")
# print("temp shape : ", temp.shape)
# print(raw_data.head())
