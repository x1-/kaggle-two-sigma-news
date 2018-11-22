
#%%
get_ipython().magic(u'matplotlib')


#%%
import _pickle as pickle
import numpy as np
import nltk 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
from datetime import date

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


#%%
import time
import warnings
warnings.filterwarnings('ignore')


#%%
repo_dir = '/Users/a12884/repos/Deep-Learning-Financial-News-Stock-Movement-Prediction/xingyou_song_event_embeddings'


#%%
with open(repo_dir + '/word_embeddings/run_info.p', 'rb') as f:
    x = pickle.load(f, encoding='bytes')


#%%
xx = dict()
for k,v in x.items():
    print(type(k), type(v))
    if isinstance(v, np.ndarray):
        xx[k.decode('utf8')] = v
    elif isinstance(v, list):
        xx[k.decode('utf8')] = map(lambda z: z.decode('utf8'), v)
    else:
        vv = dict()
        for k2,v2 in v.items():
            new_k2 = k2
            if isinstance(k2, tuple):
                new_k2 = tuple(map(lambda z: z.decode('utf8'), k2))
            elif isinstance(k2, bytes):
                new_k2 = k2.decode('utf8')
            vv[new_k2] = v2
        xx[k.decode('utf8')] = vv


#%%
xx.keys()


#%%
info2index = xx['info2index']


#%%
counter = 0
for key_ in info2index:
    print(key_)
    print(info2index[key_])
    counter += 1
    if counter == 5:
        break


#%%
event_embeddings = pd.read_csv(repo_dir + "/word_embeddings/u_epoch_500.csv", header = None)
embedding_lst = []
for row in event_embeddings.iterrows():
    index, data = row 
    temp = data.tolist()
    actual_data = [float(x) for x in temp[0].split()]
    embedding_lst.append(actual_data)


#%%
number_to_month = {"01": "Jan", "02":"Feb", "03":"Mar", "04":"Apr", "05":"May", "06": "Jun", "07":"Jul", "08":"Aug", "09":"Sep", "10":"Oct", "11":"Nov", "12":"Dec"}
def conv_num_to_string(d): #ex: conv_num_to_string('20041001') = '01-Oct-04'
    year = d[0:4]
    month = d[4:6]
    day = d[6:8]
    new = day + "-" + number_to_month[month] + "-" + year[2:4]
    return new 

def numeric_day_distance(day1, day2): #'20140111', '20150115'
    d0 = date(int(day1[0:4]), int(day1[4:6]), int(day1[6:8]) )
    d1 = date(int(day2[0:4]), int(day2[4:6]), int(day2[6:8]) )
    delta = d0 - d1 
    return abs(delta.days)


#%%
stock_to_events = {}
for key_ in info2index:
    stock_ = key_[0]
    embedding_to_index = info2index[key_]
    date_ = key_[1]
    event = key_[2]
    new_value = [date_, embedding_to_index]
    if stock_ in stock_to_events: 
        stock_to_events[stock_].append(new_value)
    else:
        stock_to_events[stock_] = [new_value]

for stock_ in stock_to_events:
    stock_to_events[stock_] = sorted( stock_to_events[stock_], key = lambda x: x[0]  )


#%%
stock_to_events.keys()


#%%
stock_to_events['GOOGL'][:5]


#%%
stk_lst_0 = []

for stock in stock_to_events:
    stock_length = {}
    stock_length["Number of Articles"] = len(stock_to_events[stock])
    stock_length["Stock Name"] = stock
    stk_lst_0.append(stock_length)


#%%
df = pd.DataFrame(stk_lst_0)
df[ ["Stock Name", "Number of Articles"]]


#%%
news_csv = pd.read_csv(repo_dir + "/news_data/news_reuters_10.csv", error_bad_lines=False, header = None, names = ["stock", "company", "date", "title", "summary", "type", "website"])
google_price_csv = pd.read_csv(repo_dir + "/price_data/GOOGL_2006-01-01_to_2017-11-01.csv")


#%%
def up_down_ratio(stock, day_lag): #ex: sentiment_to_price_plot("AAPL", 1, 'neg')
    stock_data = news_csv[news_csv["stock"] == stock]
    stock_price_csv = pd.read_csv(repo_dir + "/price_data/"+ stock+"_2006-01-01_to_2017-11-01.csv")
    total = []
    for index, row in stock_data.iterrows():
    

        day = conv_num_to_string(str(row["date"]) )

        if day in stock_price_csv["Date"].values:

            

            row_index = stock_price_csv.index[stock_price_csv["Date"] == day].tolist()[0]
            next_price = stock_price_csv.iloc[row_index - day_lag  ]
            #print next_price["Date"], google_price_csv.iloc[row_index]["Date"]
            diff = next_price["Close"] - next_price["Open"]
            if diff >= 0.0:
                total.append(1) 
            else:
                total.append(0)
    return 100*sum(total)/len(total)


#%%
get_ipython().run_cell_magic(u'time', u'', u'stk_lst = []\n\nfor stock in stock_to_events:\n    if stock != \'IBM\':\n        stock_length = {}\n        stock_length["Price Up Percentage"] = up_down_ratio(stock, 1)\n        stock_length["Stock Name"] = stock\n        stk_lst.append(stock_length)')


#%%
df = pd.DataFrame(stk_lst)
df[["Stock Name", "Price Up Percentage"]]


#%%
def total_embedding_to_class(stock, training_ratio, shuff_bool): 


    start_time = time.time()

    #----------------------------LAGS
    day_lag = 1 
    week_lag = 7
    month_lag = 30
    #--------------------------- NN parameters
    input_size = 100
    window_size_convM =3
    hidden_size_convM = 20

    window_size_convL =3 
    hidden_size_convL = 40


    hidden_size_end = 200

    learning_rate = 0.001
    num_epochs = 500
    batch_size = 50

    #training_ratio = 0.8
    #stock = 'AAPL'

    #----------------------------------- PARAMTETRS


    stock_price_csv = pd.read_csv(repo_dir + "/price_data/"+ stock+"_2006-01-01_to_2017-11-01.csv")

    temp_x = []
    temp_y = []
    for i in range(len(stock_to_events[stock]) ):
        event = stock_to_events[stock][i]


        date_numeric = event[0]
        day = conv_num_to_string( date_numeric    )

        if day in stock_price_csv["Date"].values:

            temp = {}
            temp["day"] = embedding_lst[event[1]]



            row_index = stock_price_csv.index[stock_price_csv["Date"] == day].tolist()[0]
            next_price_data = stock_price_csv.iloc[row_index - day_lag  ]

            next_price = next_price_data["Close"] - next_price_data["Open"]

            if next_price >= 0.0:
                pos_neg_class = 1
            else:
                pos_neg_class = 0
            temp_y.append(pos_neg_class)



            temp["week"] = []
            temp_date_before = 1
            while ( i - temp_date_before >= 0 and numeric_day_distance(stock_to_events[stock][i-temp_date_before][0], date_numeric ) <= week_lag ) :
                temp['week'].append(embedding_lst[stock_to_events[stock][i-temp_date_before][1]]    )
                temp_date_before +=1 

            temp["month"] = []
            temp_date_before = 1
            while ( i - temp_date_before >= 0 and numeric_day_distance(stock_to_events[stock][i-temp_date_before][0], date_numeric ) <= month_lag ) :
                temp['month'].append(embedding_lst[stock_to_events[stock][i-temp_date_before][1]]    )
                temp_date_before +=1 
            temp_x.append(temp)
    #--------------------------------------

    print ("Generated event embedded data.")
    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



    #--------------------------
    sample_size = len(temp_x)
    cut = int(training_ratio*float(sample_size) ) 
    train_x = temp_x[0:cut ]
    test_x = temp_x[cut+1:]

    train_y = temp_y[0:cut]
    test_y = temp_y[cut+1:]


    max_event_length_week = max([len(day_embedding["week"]) for day_embedding in temp_x ])
    max_event_length_month = max([len(day_embedding["month"]) for day_embedding in temp_x ])


    train_x_concatenate = []



    for day_embedding in train_x: 
        block = [day_embedding["day"]]
        week_padding_number = max_event_length_week - len(day_embedding["week"])
        block = block + day_embedding["week"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(week_padding_number)]

        month_padding_number = max_event_length_month - len(day_embedding["month"])
        block = block + day_embedding["month"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(month_padding_number)]


        train_x_concatenate.append(block)

    train_x_concatenate = torch.FloatTensor(train_x_concatenate)

    train_y = torch.LongTensor(train_y)
    train_x_concatenate_temp = torch.utils.data.TensorDataset(train_x_concatenate, train_y)
    train_loader_total = torch.utils.data.DataLoader(dataset=train_x_concatenate_temp, 
                                           batch_size = batch_size, 
                                           shuffle=shuff_bool)



    #-----------------------------------

    test_x_concatenate = []


    for day_embedding in test_x: 
        block = [day_embedding["day"]]
        week_padding_number = max_event_length_week - len(day_embedding["week"])
        block = block + day_embedding["week"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(week_padding_number)]

        month_padding_number = max_event_length_month - len(day_embedding["month"])
        block = block + day_embedding["month"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(month_padding_number)]


        test_x_concatenate.append(block)


    test_x_concatenate = torch.FloatTensor(test_x_concatenate)

    test_y = torch.LongTensor(test_y)
    test_x_concatenate_temp = torch.utils.data.TensorDataset(test_x_concatenate, test_y)
    test_loader_total = torch.utils.data.DataLoader(dataset=test_x_concatenate_temp, 
                                           batch_size = batch_size, 
                                           shuffle=False)



    #------------------------------------------
    class Net(nn.Module):

        def __init__(self, input_size, window_size_convM, hidden_size_convM, window_size_convL, hidden_size_convL, hidden_size_end):
            super(Net, self).__init__()

            self.f1 = nn.Linear(input_size + hidden_size_convM + hidden_size_convL , hidden_size_end)
            self.sigmoid = nn.Sigmoid()
            self.f2 = nn.Linear(hidden_size_end, 2)
            self.softmax = nn.Softmax()

            self.convM = nn.Conv1d(max_event_length_week, hidden_size_convM, window_size_convM, padding = 1 )
            self.poolM = nn.MaxPool1d(input_size)

            self.convL = nn.Conv1d(max_event_length_month, hidden_size_convL, window_size_convL, padding = 1 )
            self.poolL = nn.MaxPool1d(input_size)

        def forward(self, giant_block):

            S = giant_block[:, 0,]

            M = giant_block[:,1: max_event_length_week+1,].contiguous()

            L = giant_block[:,max_event_length_week+1:,].contiguous()

            #--------------------LARGE

            out_L = self.convL(L)
            out_L = self.poolL(out_L)
            out_L = out_L.view(-1, hidden_size_convL)
            #-------------------LARGE

            #------------------- MIDDLE
            out_M = self.convM(M)
            out_M = self.poolM(out_M)
            out_M = out_M.view(-1, hidden_size_convM)
            #-------------------MIDDLE


            #x = concatenation S, M, L
            x = torch.cat((out_L, out_M,S, ), 1) 

            out = self.f1(x)
            out = self.sigmoid(out)
            out = self.f2(out)
            out = self.sigmoid(out) 
            out = self.softmax(out)
            return out 

    net = Net(input_size, window_size_convM, hidden_size_convM, window_size_convL, hidden_size_convL, hidden_size_end )

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        for i, (inp, outp) in enumerate(train_loader_total):

            inp = Variable(inp, requires_grad=True)
            outp = Variable(outp)

            optimizer.zero_grad()
            outputs = net(inp)
            loss = criterion(outputs, outp.squeeze()  )
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            elapsed_time = time.time() - start_time
            print ("epoch:{0} elapsed_time:{1}".format(epoch, elapsed_time) + "[sec]")


    correct = 0
    total = 0
    for inp, lab in train_loader_total:
        inp = Variable(inp)

        outputs = net(inp)
        _, predicted = torch.max(outputs.data, 1)
        total += lab.size(0)
        correct += (predicted == lab).sum()


    train_accuracy = float(correct)/total 


    #--------------------------------------------------
    elapsed_time = time.time() - start_time
    print ("train_accuracy, elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #--------------------------------------------------

    correct = 0
    total = 0
    for inp, lab in test_loader_total:
        inp = Variable(inp, requires_grad=True )

        outputs = net(inp)
        _, predicted = torch.max(outputs.data, 1)
        total += lab.size(0)
        correct += (predicted == lab).sum()


    test_accuracy = float(correct)/total 

    return (str(train_accuracy)[0:5], str(test_accuracy)[0:5] )


#%%
get_ipython().run_cell_magic(u'time', u'', u"total_embedding_to_class('AAPL', 0.7, True)")


#%%
population = ['GOOGL', 'INTC', 'AAPL', 'CSCO', 'QCOM', 'NVDA', 'AMZN', 'MSFT']
stk_lst = []

training_ratio_population = [0.1*float(i) for i in range(5,10)]
training_ratio_names = [str(0.1*float(i)) + " Training Ratio" for i in range(5,10) ]
for stock in population:
    print (stock)
    stock_length = {}
    for j in range(len(training_ratio_population)):
        stock_length[training_ratio_names[j]] = total_embedding_to_class(stock, training_ratio_population[j], True)
    stock_length["Stock Name"] = stock
    stk_lst.append(stock_length)


#%%
df = pd.DataFrame(stk_lst)
df[["Stock Name"] + training_ratio_names  ]


#%%
def debug_total_embedding_to_class(stock, training_ratio, shuff_bool): 


    start_time = time.time()

    #----------------------------LAGS
    day_lag = 1 
    week_lag = 7
    month_lag = 30
    #--------------------------- NN parameters
    input_size = 100
    window_size_convM =3
    hidden_size_convM = 20

    window_size_convL =3 
    hidden_size_convL = 40


    hidden_size_end = 200

    learning_rate = 0.001
    #num_epochs = 500
    num_epochs = 10
    #batch_size = 50
    batch_size = 10

    #training_ratio = 0.8
    #stock = 'AAPL'

    #----------------------------------- PARAMTETRS


    stock_price_csv = pd.read_csv(repo_dir + "/price_data/"+ stock+"_2006-01-01_to_2017-11-01.csv")

    temp_x = []
    temp_y = []
    for i in range(len(stock_to_events[stock]) ):
        event = stock_to_events[stock][i]


        date_numeric = event[0]
        day = conv_num_to_string( date_numeric    )


        if day in stock_price_csv["Date"].values:

            temp = {}
            temp["day"] = embedding_lst[event[1]]



            row_index = stock_price_csv.index[stock_price_csv["Date"] == day].tolist()[0]
            next_price_data = stock_price_csv.iloc[row_index - day_lag  ]

            next_price = next_price_data["Close"] - next_price_data["Open"]

            if next_price >= 0.0:
                pos_neg_class = 1
            else:
                pos_neg_class = 0
            temp_y.append(pos_neg_class)



            temp["week"] = []
            temp_date_before = 1
            while ( i - temp_date_before >= 0 and numeric_day_distance(stock_to_events[stock][i-temp_date_before][0], date_numeric ) <= week_lag ) :
                temp['week'].append(embedding_lst[stock_to_events[stock][i-temp_date_before][1]]    )
                temp_date_before +=1 

            temp["month"] = []
            temp_date_before = 1
            while ( i - temp_date_before >= 0 and numeric_day_distance(stock_to_events[stock][i-temp_date_before][0], date_numeric ) <= month_lag ) :
                temp['month'].append(embedding_lst[stock_to_events[stock][i-temp_date_before][1]]    )
                temp_date_before +=1 
            temp_x.append(temp)
    #--------------------------------------

    print ("Generated event embedded data.")
    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print ("X")
    print (temp_x[:5])
    print ("Y")
    print (temp_y[:5])

    #--------------------------
    sample_size = len(temp_x)
    cut = int(training_ratio*float(sample_size) ) 
    train_x = temp_x[0:cut ]
    test_x = temp_x[cut+1:]

    train_y = temp_y[0:cut]
    test_y = temp_y[cut+1:]


    max_event_length_week = max([len(day_embedding["week"]) for day_embedding in temp_x ])
    max_event_length_month = max([len(day_embedding["month"]) for day_embedding in temp_x ])

    print ("max_event_length_week: {0}".format(max_event_length_week))
    print ("max_event_length_month: {0}".format(max_event_length_month))

    train_x_concatenate = []



    for day_embedding in train_x: 
        block = [day_embedding["day"]]
        week_padding_number = max_event_length_week - len(day_embedding["week"])
        block = block + day_embedding["week"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(week_padding_number)]

        month_padding_number = max_event_length_month - len(day_embedding["month"])
        block = block + day_embedding["month"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(month_padding_number)]


        train_x_concatenate.append(block)

    train_x_concatenate = torch.FloatTensor(train_x_concatenate)

    train_y = torch.LongTensor(train_y)
    train_x_concatenate_temp = torch.utils.data.TensorDataset(train_x_concatenate, train_y)
    train_loader_total = torch.utils.data.DataLoader(dataset=train_x_concatenate_temp, 
                                           batch_size = batch_size, 
                                           shuffle=shuff_bool)



    #-----------------------------------

    test_x_concatenate = []


    for day_embedding in test_x: 
        block = [day_embedding["day"]]
        week_padding_number = max_event_length_week - len(day_embedding["week"])
        block = block + day_embedding["week"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(week_padding_number)]

        month_padding_number = max_event_length_month - len(day_embedding["month"])
        block = block + day_embedding["month"]
        block = block + [[0.0 for i in range(input_size) ] for j in range(month_padding_number)]


        test_x_concatenate.append(block)


    test_x_concatenate = torch.FloatTensor(test_x_concatenate)

    test_y = torch.LongTensor(test_y)
    test_x_concatenate_temp = torch.utils.data.TensorDataset(test_x_concatenate, test_y)
    test_loader_total = torch.utils.data.DataLoader(dataset=test_x_concatenate_temp, 
                                           batch_size = batch_size, 
                                           shuffle=False)



    #------------------------------------------
    class Net(nn.Module):

        def __init__(self, input_size, window_size_convM, hidden_size_convM, window_size_convL, hidden_size_convL, hidden_size_end):
            super(Net, self).__init__()

            self.f1 = nn.Linear(input_size + hidden_size_convM + hidden_size_convL , hidden_size_end)
            self.sigmoid = nn.Sigmoid()
            self.f2 = nn.Linear(hidden_size_end, 2)
            self.softmax = nn.Softmax()

            self.convM = nn.Conv1d(max_event_length_week, hidden_size_convM, window_size_convM, padding = 1 )
            self.poolM = nn.MaxPool1d(input_size)

            self.convL = nn.Conv1d(max_event_length_month, hidden_size_convL, window_size_convL, padding = 1 )
            self.poolL = nn.MaxPool1d(input_size)

        def forward(self, giant_block):

            S = giant_block[:, 0,]

            M = giant_block[:,1: max_event_length_week+1,].contiguous()

            L = giant_block[:,max_event_length_week+1:,].contiguous()

            #--------------------LARGE

            out_L = self.convL(L)
            out_L = self.poolL(out_L)
            out_L = out_L.view(-1, hidden_size_convL)
            #-------------------LARGE

            #------------------- MIDDLE
            out_M = self.convM(M)
            out_M = self.poolM(out_M)
            out_M = out_M.view(-1, hidden_size_convM)
            #-------------------MIDDLE


            #x = concatenation S, M, L
            x = torch.cat((out_L, out_M,S, ), 1) 

            out = self.f1(x)
            out = self.sigmoid(out)
            out = self.f2(out)
            out = self.sigmoid(out) 
            out = self.softmax(out)
            return out 

    net = Net(input_size, window_size_convM, hidden_size_convM, window_size_convL, hidden_size_convL, hidden_size_end )

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        for i, (inp, outp) in enumerate(train_loader_total):

            inp = Variable(inp, requires_grad=True)
            outp = Variable(outp)

            optimizer.zero_grad()
            outputs = net(inp)
            loss = criterion(outputs, outp.squeeze()  )
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            elapsed_time = time.time() - start_time
            print ("epoch:{0} elapsed_time:{1}".format(epoch, elapsed_time) + "[sec]")


    correct = 0
    total = 0
    for inp, lab in train_loader_total:
        inp = Variable(inp)

        outputs = net(inp)
        _, predicted = torch.max(outputs.data, 1)
        total += lab.size(0)
        correct += (predicted == lab).sum()


    train_accuracy = float(correct)/total 


    #--------------------------------------------------
    elapsed_time = time.time() - start_time
    print ("train_accuracy, elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #--------------------------------------------------

    correct = 0
    total = 0
    for inp, lab in test_loader_total:
        inp = Variable(inp, requires_grad=True )

        outputs = net(inp)
        _, predicted = torch.max(outputs.data, 1)
        total += lab.size(0)
        correct += (predicted == lab).sum()


    test_accuracy = float(correct)/total 

    return (str(train_accuracy)[0:5], str(test_accuracy)[0:5] )


#%%
debug_total_embedding_to_class('AAPL', 0.7, True)


