import matplotlib
import numpy as np
import mnist
import pickle
from hrvanalysis import get_time_domain_features,get_frequency_domain_features,get_sampen
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pyrebase
import random
from keras import optimizers
import peakutils


from sklearn.ensemble import ExtraTreesClassifier
import cvxEDA
import pandas as pd
import matplotlib
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier , XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import seaborn as sns;
from elm import ELM

import pyhrv.frequency_domain as fd
import biosppy
from sklearn.datasets import make_moons, make_circles, make_classification
import biosppy.signals.tools as signalsTools

from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.datasets import fetch_openml as fetch_mldata

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import heartpy as hp
import sampen
import neurokit
import json
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
from sampen import sampen2
import numpy as np
import elm


config = {
    "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
    "authDomain": "synapsys-388ba.firebaseapp.com",
    "databaseURL": "https://synapsys-388ba.firebaseio.com/",
    "storageBucket": "synapsys-388ba.appspot.com"
}

eliminate_time=30
chunk_time=30
def eliminate_x_tail(array,x_sec):
    # time = array[-1]['timeStamp'] - array[0]['timeStamp']
    first_time = array[0]['timeStamp']
    cut_from = 0
    for node in array:
        cut_from += 1
        if node['timeStamp']-first_time > x_sec:
            break
    return array[cut_from:len(array)]


def split_signal_to_x_sec_chunks(array,x_sec):
    first_time = array[0]['timeStamp']
    time = array[-1]['timeStamp']-array[0]['timeStamp']
    chunks_number = round(time/x_sec)
    signal_chunks = [None] * chunks_number
    for node in array:
        chunk_n = int((node['timeStamp'] - first_time)/x_sec)
        if chunk_n < len(signal_chunks):
            if not signal_chunks[chunk_n]:
                temp_arr = [node]
                signal_chunks[chunk_n] = temp_arr
            else:
                signal_chunks[chunk_n].extend([node])

    return signal_chunks

def get_apen(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def bvpPeaks(signal):
    cb = np.array(signal)
    x = peakutils.indexes(cb, thres=0.02/max(cb), min_dist=0.1)
    y = []
    i = 0
    while (i < (len(x)-1)):
        if x[i+1] - x[i] < 15:
            y.append(x[i])
            x = np.delete(x, i+1)
        else:
            y.append(x[i])
        i += 1
    return y

def getRRI(signal, start, sample_rate):
    peakIDX = bvpPeaks(signal)
    spr = 1 / sample_rate # seconds between readings
    start_time = float(start)
    timestamp = [start_time, (peakIDX[0] * spr) + start_time ]
    ibi = [0, 0]
    for i in range(1, len(peakIDX)):
        timestamp.append(peakIDX[i] * spr + start_time)
        ibi.append((peakIDX[i] - peakIDX[i-1]) * spr)

    df = pd.DataFrame({'Timestamp': timestamp, 'IBI': ibi})
    return df

def getHRV(data, avg_heart_rate):
    rri = np.array(data['IBI']) * 1000
    RR_list = rri.tolist()
    #RR_diff = []
    RR_sqdiff = []
    RR_diff_timestamp = []
    cnt = 2
    while (cnt < (len(RR_list)-1)):
        #RR_diff.append(abs(RR_list[cnt+1] - RR_list[cnt]))
        RR_sqdiff.append(np.math.pow(RR_list[cnt + 1] - RR_list[cnt], 2))
        RR_diff_timestamp.append(data['Timestamp'][cnt])
        cnt += 1
    hrv_window_length = 10
    window_length_samples = int(hrv_window_length*(avg_heart_rate/60))
    #SDNN = []
    RMSSD = []
    index = 1
    for val in RR_sqdiff:
        if index < int(window_length_samples):
            #SDNNchunk = RR_diff[:index:]
            RMSSDchunk = RR_sqdiff[:index:]
        else:
            #SDNNchunk = RR_diff[(index-window_length_samples):index:]
            RMSSDchunk = RR_sqdiff[(index-window_length_samples):index:]
        #SDNN.append(np.std(SDNNchunk))
        RMSSD.append(np.math.sqrt(np.std(RMSSDchunk)))
        index += 1
    dt = np.dtype('Float64')
    #SDNN = np.array(SDNN, dtype=dt)
    RMSSD = np.array(RMSSD, dtype=dt)
    df = pd.DataFrame({'Timestamp': RR_diff_timestamp, 'HRV': RMSSD})
    return df
def plot_two_signals(s1,s2,t):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, s1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, s2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def calculate_bvp_f(bvp_data, sample_rate,bvp_time,bvp_chunks):
    features_chunks = []

    for chunk in range(len(bvp_chunks)):
        if bvp_chunks[chunk]== None:
            features_chunks.extend([None])
            continue

        bvpData = list(map(lambda x: x['data'], bvp_chunks[chunk]))
        chunk_time = bvp_chunks[chunk][-1]['timeStamp']-bvp_chunks[chunk][0]['timeStamp']
        chunk_s_r=len(bvpData)/chunk_time
        if not chunk_s_r + 30 >= sample_rate :
            features_chunks.extend([None])
            continue

        bandpass = signalsTools.filter_signal(ftype='FIR', sampling_rate=chunk_s_r, band='bandpass',
                                              frequency=[0.5, 4],
                                              signal=bvpData, order=4)
        # all_working_data, all_measures = hp.process(bandpass[0], sample_rate=chunk_s_r,calc_freq=True)
        all_working_data, all_measures = hp.process(np.asarray(bvpData), sample_rate=chunk_s_r)
        hp.plotter(all_working_data, all_measures)
        result = biosppy.signals.bvp.bvp(signal=np.asarray(bvpData), sampling_rate=chunk_s_r, show=True)
        result = fd.welch_psd(nni=np.asarray(bvpData))
        # RRI_DF = getRRI(np.asarray(bvpData), column2, sample_rate)
        # HRV_DF = getHRV(RRI_DF, np.mean(HR))
        # print(result['fft_total'])
        result.fft_plot()

        f, Pxx_den = signal.welch(np.asarray(bvpData))
        plt.semilogy(f, Pxx_den)
        plt.ylim([0.5e-3, 1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
        plt.plot(all_working_data['RR_list'])
        plt.show()
        # features = {
        #     'HR_avg': all_measures['bpm'],
        #     'NN_avg': all_measures['ibi'],
        #     'SDNN': all_measures['sdnn'],
        #     'SDSD': all_measures['sdsd'],
        #     'RMSSD': all_measures['rmssd'],
        #     'pNN20': all_measures['pnn20'],
        #     'pNN50': all_measures['pnn50'],
        #     'hrMad': all_measures['hr_mad'],
        #     'BreR': all_measures['breathingrate'],
        #     'lf': all_measures['lf'],
        #     'hf': all_measures['hf'],
        #     'lf/hf': all_measures['lf/hf']
        # }

        time_domain_features = get_time_domain_features(all_working_data['RR_list'])
        freq_domain_features = get_frequency_domain_features(all_working_data['RR_list'])
        sampen_domain_features = get_sampen(all_working_data['RR_list'])
        features = {'co_he':freq_domain_features['total_power']/(freq_domain_features['hf']+freq_domain_features['lf'])}
        features.update(time_domain_features)
        features.update(freq_domain_features)
        features.update(sampen_domain_features)
        # features.update({'ApEN':get_apen(all_working_data['RR_list'], 2, (0.2 * features['SDNN']))})
        features.update({'ApEN':get_apen(all_working_data['RR_list'], 2, (0.2 * features['sdnn']))})

        # samp_enn = sampen2(all_working_data['RR_list'])
        # features['sampEn'] = samp_enn['sampen']

        SD1 = (1 / np.sqrt(2)) * features['sdsd']  # measures the width of poincare cloud https://github.com/pickus91/HRV/blob/master/poincare.py
        SD2 = np.sqrt((2 * features['sdnn'] ** 2) - (0.5 * features['sdsd'] ** 2))  # measures the length of the poincare cloud
        features['SD1'] = SD1
        features['SD2'] = SD2
        features_chunks.extend([features])

    return features_chunks



def calculate_sc_f(scdata, sample_rate,sc_time,sc_chunks):
    features_chunks = []

    for chunk in sc_chunks:

        data = list(map(lambda x: x['data'], chunk))
        y = np.asarray(data)  # convert list to Phsio model with sanple rate 400
        yn = (y - y.mean()) / y.std()
        Fs = sample_rate
        [scr, p, scl, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / Fs)
        scr_peaks = find_peaks(scr)

        sc_avg = np.average(y)

        scl_avg = np.average(scl)
        scl_slope = np.amax(scl) - np.amin(scl)

        scr_avg = np.average(scr)
        scr_max = np.amax(scr)

        scr_peaks_number = len(scr_peaks[0])

        features = {
            'sc_avg': sc_avg,
            'scl_avg': scl_avg,
            'scl_slope': scl_slope,
            'scr_avg': scr_avg,
            'scr_max': scr_max,
            'scr_peak': scr_peaks_number
        }

        features_chunks.extend([features])

    return features_chunks


def calculate_skt_f(skt_data,smaple_rate,skt_time,skt_chunks):
    features_chunks=[]

    for chunk in skt_chunks:
        skt = list(map(lambda x: x['data'], chunk))
        skt_avg = np.average(skt)
        skt_slope = np.amax(skt) - np.amin(skt)
        skt_std = np.std(skt)
        features =  {'skt_avg': skt_avg,
                    'skt_slope': skt_slope,
                    'skt_std': skt_std }

        features_chunks.extend([features])

    return features_chunks


def calculate_panelist_features(user_data, stage):
    bvp_data = user_data[stage]['bvp']
    if len(bvp_data) == 0:
        return None
    stable_bvp = eliminate_x_tail(bvp_data, eliminate_time)
    bvp_time = stable_bvp[-1]['timeStamp'] - stable_bvp[0]['timeStamp']
    bvp_sample_rate = len(stable_bvp) / bvp_time
    bvp_chunks = split_signal_to_x_sec_chunks(stable_bvp, chunk_time)

    eda_data = user_data[stage]['eda']
    if len(eda_data) == 0:
        return None
    stable_eda = eliminate_x_tail(eda_data, eliminate_time)
    eda_time = stable_eda[-1]['timeStamp'] - stable_eda[0]['timeStamp']
    eda_sample_rate = len(stable_eda) / eda_time
    eda_chunks = split_signal_to_x_sec_chunks(stable_eda, chunk_time)


    temp_data = user_data[stage]['temp']
    if len(temp_data) == 0:
        return None
    stable_temp = eliminate_x_tail(temp_data, eliminate_time)
    temp_time = stable_temp[-1]['timeStamp'] - stable_temp[0]['timeStamp']
    temp_sample_rate = len(stable_temp) / temp_time
    temp_chunks = split_signal_to_x_sec_chunks(stable_temp, chunk_time)


    skt_features = calculate_skt_f(stable_temp,temp_sample_rate,temp_time,temp_chunks)
    bvp_features = calculate_bvp_f(stable_bvp, bvp_sample_rate,bvp_time,bvp_chunks)
    eda_features = calculate_sc_f(stable_eda, eda_sample_rate,eda_time,eda_chunks)

    combined_features_chunks=[]
    for chunk in range(len(skt_features)):
        if skt_features[chunk] and eda_features[chunk] and bvp_features[chunk]:
            temp = {}
            temp.update(skt_features[chunk])
            temp.update(bvp_features[chunk])
            temp.update(eda_features[chunk])
            combined_features_chunks.extend([temp])

    return combined_features_chunks


def _main():

    # firebase = pyrebase.initialize_app(config)

    # db = firebase.database()
    # all_users_r1= db.child("cho").child("r-1").get().val()
    # all_users_r2= db.child("cho").child("r-2").get().val()
    #
    # del all_users_r2['cho']
    # all_users=db.child("cho").child("users").get().val()
    # # all_users.update(all_users_t)
    #
    #
    # a = {'hello': 'world'}

    # with open('r-2.pickle', 'wb') as handle:
    #     pickle.dump(all_users_r2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # for user in all_users:
    #     db.child("cho").child("users")
    #     user_node = db.child("cho").child("r-1").child(user)
    #     for attr in all_users[user]:
    #         user_node.child(attr).set(all_users[user][attr])
    #         print(attr)

    # child(user).set() .remove()

    with open('users.pickle', 'rb') as handle:
        all_users = pickle.load(handle)



    users_data = []
    targets = { 'BA_S':int(0),
              'MIS_S':int(1),
              'MOS_S':int(2),
              'SES_S':int(3)

    }
    for user in all_users:
        data = get_user_data(all_users[user])
        data.update({'name': user})
        users_data.append(data)

    df = pd.DataFrame()
    excluded = pd.DataFrame()
    count=0
    excludedPanelistsN=1
    randomExcluded=4
    for user_data in users_data:
        if user_data["name"]=="sameha" or user_data["name"]=="rna":
            continue
        count+=1
        # if count==randomExcluded:
        #     df=excluded
        # else:
        #     df=included
        try:
            BA_S_features_chunks = calculate_panelist_features(user_data, 'BA_S')
            MIS_S_features_chunks = calculate_panelist_features(user_data, 'MIS_S')
            MOS_S_features_chunks = calculate_panelist_features(user_data, 'MOS_S')
            SES_S_features_chunks = calculate_panelist_features(user_data, 'SES_S')
            # RE_S_features_chunks = calculate_panelist_features(user_data, 'RE_S')

            for chunk in range(len(BA_S_features_chunks)):
                BA_S_features_chunks[chunk].update({'tar':targets['BA_S'], 'pi': user_data['name']})
                df = df.append(BA_S_features_chunks[chunk],ignore_index=True)

            for chunk in range(len(MIS_S_features_chunks)):
                MIS_S_features_chunks[chunk].update({'tar': targets['MIS_S'], 'pi': user_data['name']})
                df = df.append(MIS_S_features_chunks[chunk], ignore_index=True)

            for chunk in range(len(MOS_S_features_chunks)):
                MOS_S_features_chunks[chunk].update({'tar': targets['MOS_S'], 'pi': user_data['name']})
                df = df.append(MOS_S_features_chunks[chunk], ignore_index=True)

            for chunk in range(len(SES_S_features_chunks)):
                SES_S_features_chunks[chunk].update({'tar': targets['SES_S'], 'pi': user_data['name']})
                df = df.append(SES_S_features_chunks[chunk], ignore_index=True)
        except:
            print('err : '+user_data["name"] )

    df['sampen']=df['sampen'].apply(lambda x: x if (x<100 and x>-100)  else 0 )

    dfFeatures=df.loc[:, df.columns.difference(['tar','pi'])].copy()
    dfNoramlized=df.loc[:, df.columns.difference(['tar'])].copy().groupby('pi').transform(lambda x: (x-x.mean())/x.std() if (x.dtype == np.number) else x )
    # dfNoramlized = dfFeatures.apply(lambda x: x if x.isnumeric() else 0 , axis=0)
    # dfNoramlized = dfFeatures.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # dfNoramlized = dfFeatures.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    # dfNoramlized = dfFeatures
    # df.dropna(axis=0,inplace=True)
    # df.reset_index(inplace=True,drop=True)
    # df.drop(inplace=True, columns='index')
    # df['sampen']=df['sampen'].apply(lambda x: x if (x<100 and x>-100)  else 0 )
    # dfNoramlized = df.copy()
    dfNoramlized['tar']=df['tar']
    dfNoramlized['pi']=df['pi']
    # dfNoramlized=dfNoramlized.dropna(axis=1)
    #
    # excluded= dfNoramlized.loc[0:46].copy().dropna()
    excluded = dfNoramlized.loc[dfNoramlized['pi'].isin(['','ran','mah'])]
    # excluded.loc[excluded['tar'] == 0.0, 'tar'] = -1
    # excluded.loc[excluded['tar'] != -1.0, 'tar'] = 0
    # excluded.loc[excluded['tar'] == -1.0, 'tar'] = 1

    # included= dfNoramlized.loc[47:len(df)].copy().dropna()
    included = dfNoramlized.loc[ ~dfNoramlized['pi'].isin(['','ran','mah']) ]

    features_name = ['ApEN', 'SD1', 'SD2', 'cvnni', 'cvsd', 'hf', 'hfnu', 'lf',
                     'lf_hf_ratio', 'lfnu', 'max_hr', 'mean_hr', 'mean_nni', 'median_nni',
                     'min_hr', 'nni_20', 'nni_50', 'pnni_20', 'pnni_50', 'range_nni',
                     'rmssd', 'sampen', 'sc_avg', 'scl_avg', 'scl_slope', 'scr_avg',
                     'scr_max', 'scr_peak', 'sdnn', 'sdsd', 'skt_avg', 'skt_slope',
                     'skt_std', 'std_hr', 'total_power', 'vlf','co_he']
    features_name = ('ApEN', 'SD1', 'SD2','hf', 'lf',
                     'lf_hf_ratio', 'mean_hr',
                      'pnni_20', 'pnni_50',
                     'rmssd', 'sampen', 'sc_avg', 'scl_avg', 'scl_slope', 'scr_avg',
                     'scr_max', 'scr_peak', 'sdnn', 'sdsd', 'skt_avg', 'skt_slope',
                     'skt_std','co_he')

    includedTargets = [i if i else i for i in included['tar'].values.astype(int)]
    excludedTargets = [i if i else i for i in  excluded['tar'].values.astype(int)]
    includedFeatures = included.loc[:, features_name].copy().values
    excludedFeatures = excluded.loc[:, features_name].copy().values

    # includedFeatures = included.loc[:, included.columns.difference(['tar', 'pi'])].copy().values
    # excludedFeatures = excluded.loc[:, excluded.columns.difference(['tar', 'pi'])].copy().values

    regersors={}

    for indexTarget in range(len(set(list(included['tar'].values)))):
        # indexTarget=0
        includedTargets =[1 if i == indexTarget else 0 for i in included['tar'].values.astype(int)]
        excludedTargets =[1 if i == indexTarget else 0 for i in excluded['tar'].values.astype(int)]
        # includedFeatures = included.loc[:, included.columns.difference(['tar','pi'])].copy().values
        includedFeatures = included.loc[:, included.columns.difference(['tar','pi'])].copy().values
        # excludedFeatures = excluded.loc[:, excluded.columns.difference(['tar','pi'])].copy().values
        excludedFeatures = excluded.loc[:, excluded.columns.difference(['tar','pi'])].copy().values


        model = ExtraTreesClassifier( max_depth=1000,)
        model.fit(includedFeatures, includedTargets)

        y_pred = model.predict_proba(excludedFeatures)
        predictions = [round(value[1]) for value in y_pred]
        accuracy = accuracy_score(excludedTargets, predictions)
        print('accuracy :' + str(accuracy))

        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(includedFeatures.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(includedFeatures.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(includedFeatures.shape[1]), indices)
        plt.xlim([-1, includedFeatures.shape[1]])
        plt.show()

        model = Sequential([
            Dense(200, activation='relu', input_shape=(len(includedFeatures[0]),)),
            Dense(150, activation='relu'),
            Dense(100, activation='relu'),
            Dense(20, activation='relu'),
            Dense(2, activation='softmax'),
        ])
        # from keras import optimizers

        # Compile the model.
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Train the model.
        model.fit(
            includedFeatures,
            to_categorical(includedTargets),
            epochs=200,
            batch_size=200,
        )
        accuracy = model.evaluate(
            excludedFeatures,
            to_categorical(excludedTargets)
        )
        y_pred = model.predict_proba(excludedFeatures)
        predictions = [round(value[1]) for value in y_pred]
        # accuracy = accuracy_score(excludedTargets, predictions)
        print('accuracy :'+str(accuracy))
        regersors[indexTarget]={'model':model,'accuracy':accuracy}
    excludedTargets = [i if i  else i for i in excluded['tar'].values.astype(int)]
    dfTest=pd.DataFrame()
    dfTest['test']=excludedTargets
    for indexTarget in range(len(regersors)):
        y_pred = regersors[indexTarget]['model'].predict_proba(excludedFeatures)
        dfTest[indexTarget]=[val[1]for val in y_pred]

    finalTest=[]
    for index,row in dfTest.iterrows():
        maxIndex=-1
        maxVal=-1
        for indexTarget in range(len(regersors)):
            if row[indexTarget]>maxVal:
                maxVal=row[indexTarget]
                maxIndex=indexTarget

        finalTest.append(maxIndex)

    accuracy = accuracy_score(excludedTargets, finalTest)
    print("final excluded Accuracy: %.2f%%" % (accuracy * 100.0))
    pyplot.bar(included.columns.difference(['tar', 'pi']), model.feature_importances_)
    pyplot.show()


    model = Sequential([
        Dense(500, activation='relu', input_shape=(len(includedFeatures[0]),)),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(20, activation='relu'),
        Dense(4, activation='softmax'),
    ])
    # from keras import optimizers

    # Compile the model.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
        includedFeatures,
        to_categorical(includedTargets),
        epochs=200,
        batch_size=200,
    )
    model.evaluate(
        excludedFeatures,
        to_categorical(excludedTargets)
    )
    # included.loc[included['tar'] == 0.0, 'tar'] = -1
    # included.loc[included['tar'] != -1.0, 'tar'] = 0
    # included.loc[included['tar'] == -1.0, 'tar'] = 1
    # regersors={}
    # for indexTarget in range(len(set(list(included['tar'].values)))):
    #     try:
    #         # indexTarget=0
    #         includedTargets =[1 if i == indexTarget else 0 for i in included['tar'].values.astype(int)]
    #         excludedTargets =[1 if i == indexTarget else 0 for i in excluded['tar'].values.astype(int)]
    #         includedFeatures = included.loc[:, included.columns.difference(['tar','pi'])].copy().values
    #         excludedFeatures = excluded.loc[:, excluded.columns.difference(['tar','pi'])].copy().values
    #


    #         # Compile the model.
    #         model = Sequential([
    #             Dense(220, activation='relu', input_shape=(22,)),
    #             Dense(120, activation='relu'),
    #             Dense(100, activation='relu'),
    #             Dense(20, activation='relu'),
    #             Dense(2, activation='softmax'),
    #         ])
    #         model.compile(
    #             optimizer='adam',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'],
    #         )
    #
    #         # Train the model.
    #         model.fit(
    #             includedFeatures,
    #             to_categorical(includedTargets),
    #             epochs=200,
    #             batch_size=32,
    #         )
    #         model.evaluate(
    #             excludedFeatures,
    #             to_categorical(excludedTargets)
    #         )
    #         y_pred = model.predict(excludedFeatures)
    #         predictions = [round(value[1]) for value in y_pred]
    #         accuracy = accuracy_score(excludedTargets, predictions)
    #         regersors[indexTarget]={'model':model,'accuracy':accuracy}
    #     except:
    #         print("An exception occurred")
    #
    #
    # excludedTargets = [i if i  else i for i in excluded['tar'].values.astype(int)]
    # dfTest=pd.DataFrame()
    # dfTest['test']=excludedTargets
    # for indexTarget in range(len(regersors)):
    #     y_pred = regersors[indexTarget]['model'].predict(excludedFeatures)
    #     dfTest[indexTarget]=[val[1]for val in y_pred]
    #
    # finalTest=[]
    # for index,row in dfTest.iterrows():
    #     maxIndex=-1
    #     maxVal=-1
    #     for indexTarget in range(len(regersors)):
    #         if row[indexTarget]>maxVal:
    #             maxVal=row[indexTarget]
    #             maxIndex=indexTarget
    #
    #     finalTest.append(maxIndex)
    #
    # accuracy = accuracy_score(excludedTargets, finalTest)
    # print("final excluded Accuracy: %.2f%%" % (accuracy * 100.0))



def get_user_data(user):
    panelist = {}
    panelist['BA-SEnd'] = user['BA-SEnd']
    panelist['BA-SStart'] = user['BA-SStart']
    panelist['MIS-SEnd'] = user['MIS-SEnd']
    panelist['MIS-SStart'] = user['MIS-SStart']
    panelist['MOS-SEnd'] = user['MOS-SEnd']
    panelist['MOS-SStart'] = user['MOS-SStart']
    panelist['SES-SStart'] = user['SES-SStart']
    panelist['SES-SEnd']=user['SES-SEnd']

    # panelist['RE-SEnd'] = user['RE-SEnd']
    # panelist['RE-SStart'] = user['RE-SStart']
    ba_s_bvpdata = []
    mis_s_bvpdata = []
    mos_s_bvpdata = []
    ses_s_bvpdata = []
    re_s_bvpdata = []


    for (k, v) in user['bvpData'].items():
        if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
            ba_s_bvpdata.append(v)
        if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MIS-SEnd'] / 1000):
            mis_s_bvpdata.append(v)
        if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MOS-SEnd'] / 1000):
            mos_s_bvpdata.append(v)
        if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['SES-SEnd'] / 1000):
            ses_s_bvpdata.append(v)
        # if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
        #         panelist['RE-SEnd'] / 1000):
        #     re_s_bvpdata.append(v)

    ba_s_gsrdata = []
    mis_s_gsrdata = []
    mos_s_gsrdata = []
    ses_s_gsrdata = []
    re_s_gsrdata = []
    for (k, v) in user['gsrData'].items():
        if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
            ba_s_gsrdata.append(v)
        if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MIS-SEnd'] / 1000):
            mis_s_gsrdata.append(v)
        if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MOS-SEnd'] / 1000):
            mos_s_gsrdata.append(v)
        if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['SES-SEnd'] / 1000):
            ses_s_gsrdata.append(v)
        # if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
        #         panelist['RE-SEnd'] / 1000):
        #     re_s_gsrdata.append(v)

    # ba_s_ibidata = []
    # mis_s_ibidata = []
    # mos_s_ibidata = []
    # ses_s_ibidata = []
    # re_s_ibidata = []
    #
    # for (k, v) in user['ibiData'].items():
    #     if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
    #         ba_s_ibidata.append(v)
    #     if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(
    #             panelist['MIS-SEnd'] / 1000):
    #         mis_s_ibidata.append(v)
    #     if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(
    #             panelist['MOS-SEnd'] / 1000):
    #         mos_s_ibidata.append(v)
    #     if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(
    #             panelist['SES-SEnd'] / 1000):
    #         ses_s_ibidata.append(v)
    #     if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
    #             panelist['RE-SEnd'] / 1000):
    #         re_s_ibidata.append(v)

    ba_s_tempdata = []
    mis_s_tempdata = []
    mos_s_tempdata = []
    ses_s_tempdata = []
    re_s_tempdata = []

    for (k, v) in user['tempData'].items():
        if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
            ba_s_tempdata.append(v)
        if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MIS-SEnd'] / 1000):
            mis_s_tempdata.append(v)
        if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MOS-SEnd'] / 1000):
            mos_s_tempdata.append(v)
        if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['SES-SEnd'] / 1000):
            ses_s_tempdata.append(v)
        # if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
        #         panelist['RE-SEnd'] / 1000):
        #     re_s_tempdata.append(v)

    data = {
        'BA_S': {
            'bvp': ba_s_bvpdata,
            'eda': ba_s_gsrdata,
            'temp': ba_s_tempdata
        },
        'MIS_S': {
            'bvp': mis_s_bvpdata,
            'eda': mis_s_gsrdata,
            'temp': mis_s_tempdata
        },
        'MOS_S': {
            'bvp': mos_s_bvpdata,
            'eda': mos_s_gsrdata,
            'temp': mos_s_tempdata
        },
        'SES_S': {
            'bvp': ses_s_bvpdata,
            'eda': ses_s_gsrdata,
            'temp': ses_s_tempdata
        },
        'RE_S': {
                        'bvp': re_s_bvpdata,
                        'eda': re_s_gsrdata,
                        'temp': re_s_tempdata
        },

    }
    return data





_main()
