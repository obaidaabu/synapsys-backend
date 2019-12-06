import matplotlib

import pyrebase
import cvxEDA
import pandas as pd
import matplotlib

import seaborn as sns;
from elm import ELM


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
from sampen import sampen2
import numpy as np
import elm


config = {
    "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
    "authDomain": "synapsys-388ba.firebaseapp.com",
    "databaseURL": "https://synapsys-388ba.firebaseio.com/",
    "storageBucket": "synapsys-388ba.appspot.com"
}

eliminate_time=0
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
                                              signal=bvpData, order=2)
        all_working_data, all_measures = hp.process(bandpass[0], sample_rate=chunk_s_r, calc_freq=True)
        features = {
            'HR_avg': all_measures['bpm'],
            'NN_avg': all_measures['ibi'],
            'SDNN': all_measures['sdnn'],
            'SDSD': all_measures['sdsd'],
            'RMSSD': all_measures['rmssd'],
            'pNN20': all_measures['pnn20'],
            'pNN50': all_measures['pnn50'],
            'lf': all_measures['lf'],
            'hf': all_measures['hf'],
            'lf/hf': all_measures['lf/hf']
        }
        features['ApEN'] = get_apen(all_working_data['RR_list_cor'], 2, (0.2 * features['SDNN']))
        # samp_enn = sampen2([]all_working_data['RR_list_cor'])
        # features['sampEn'] = samp_enn['sampen']

        SD1 = (1 / np.sqrt(2)) * features['SDSD']  # measures the width of poincare cloud https://github.com/pickus91/HRV/blob/master/poincare.py
        SD2 = np.sqrt((2 * features['SDNN'] ** 2) - (0.5 * features['SDSD'] ** 2))  # measures the length of the poincare cloud
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

    firebase = pyrebase.initialize_app(config)

    db = firebase.database()
    all_users = db.child("cho").child("users").get().val()

    users_data = []
    targets = { 'BA_S':int(0),
              'MIS_S':int(1),
              'MOS_S':int(2),
              'SES_S':int(3)

    }
    for user in all_users:
        #user='a'
        data = get_user_data(all_users[user])
        data.update({'name': user})
        users_data.append(data)

    for users_data in users_data:
        BA_S_features_chunks = calculate_panelist_features(users_data, 'BA_S')
        MIS_S_features_chunks = calculate_panelist_features(users_data, 'MIS_S')
        MOS_S_features_chunks = calculate_panelist_features(users_data, 'MOS_S')
        SES_S_features_chunks = calculate_panelist_features(users_data, 'SES_S')
        RE_S_features_chunks = calculate_panelist_features(users_data, 'RE_S')

        df = pd.DataFrame()
        for chunk in range(len(BA_S_features_chunks)):
            BA_S_features_chunks[chunk].update({'tar':targets['BA_S']})
            df = df.append(BA_S_features_chunks[chunk],ignore_index=True)

        for chunk in range(len(MIS_S_features_chunks)):
            BA_S_features_chunks[chunk].update({'tar': targets['MIS_S']})
            df = df.append(BA_S_features_chunks[chunk], ignore_index=True)

        for chunk in range(len(MOS_S_features_chunks)):
            BA_S_features_chunks[chunk].update({'tar': targets['MOS_S']})
            df = df.append(BA_S_features_chunks[chunk], ignore_index=True)

    #


    data=normalize(df.loc[:, df.columns != 'tar'].copy().values)
    # sns.pairplot(df, hue='tar');
    data= df.loc[:, df.columns != 'tar'].copy().values
    targets =[i if i == 1 else i for i in df['tar'].values.astype(int)]
    X_train, X_test, y_train, y_test = train_test_split(data,targets,test_size=0.4)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix

    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)

    # accuracy on X_test
    accuracy = gnb.score(X_test, y_test)
    print
    accuracy

    # creating a confusion matrix
    cm = confusion_matrix(y_test, gnb_predictions)

    print("ELM Accuracy %0.3f " % elm.score(X_test, y_test))
    hid_nums = [1,2,3,4,5,6,7,10, 20, 30]

    target_l = LabelEncoder().fit_transform(targets)

    for hid_num in hid_nums:
        print(hid_num, end=' ')
        e = ELM(hid_num)

        ave = 0
        for i in range(10):
            cv = KFold(n_splits=5, shuffle=True)
            scores = cross_val_score(e, data, target_l, cv=cv, scoring='accuracy', n_jobs=-1)
            ave += scores.mean()

        ave /= 10

        print("Accuracy: %0.3f " % (ave))


def get_user_data(user):
    panelist = {}
    panelist['BA-SEnd'] = user['BA-SEnd']
    panelist['BA-SStart'] = user['BA-SStart']
    panelist['MIS-SEnd'] = user['MIS-SEnd']
    panelist['MIS-SStart'] = user['MIS-SStart']
    panelist['MOS-SEnd'] = user['MOS-SEnd']
    panelist['MOS-SStart'] = user['MOS-SStart']
    panelist['SES-SStart'] = user['SES-SStart']
    if 'SES-SEnd' in user:
        panelist['SES-SEnd']=user['SES-SEnd']
    else:
        panelist['SES-SEnd'] = 1554307848588
    panelist['RE-SEnd'] = user['RE-SEnd']
    panelist['RE-SStart'] = user['RE-SStart']
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
        if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['RE-SEnd'] / 1000):
            re_s_bvpdata.append(v)

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
        if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['RE-SEnd'] / 1000):
            re_s_gsrdata.append(v)

    ba_s_ibidata = []
    mis_s_ibidata = []
    mos_s_ibidata = []
    ses_s_ibidata = []
    re_s_ibidata = []

    for (k, v) in user['ibiData'].items():
        if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
            ba_s_ibidata.append(v)
        if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MIS-SEnd'] / 1000):
            mis_s_ibidata.append(v)
        if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['MOS-SEnd'] / 1000):
            mos_s_ibidata.append(v)
        if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['SES-SEnd'] / 1000):
            ses_s_ibidata.append(v)
        if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['RE-SEnd'] / 1000):
            re_s_ibidata.append(v)

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
        if v['timeStamp'] > float(panelist['RE-SStart'] / 1000) and v['timeStamp'] < float(
                panelist['RE-SEnd'] / 1000):
            re_s_tempdata.append(v)

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
