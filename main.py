import pyrebase
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import heartpy as hp
import sampen
import neurokit
from scipy.signal import find_peaks
from sampen import sampen2
import numpy as np

config = {
  "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
  "authDomain": "synapsys-388ba.firebaseapp.com",
  "databaseURL": "https://synapsys-388ba.firebaseio.com/",
  "storageBucket": "synapsys-388ba.appspot.com"
}


def ApEn(U, m, r):
  def _maxdist(x_i, x_j):
    return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

  def _phi(m):
    x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
    C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
    return (N - m + 1.0) ** (-1) * sum(np.log(C))

  N = len(U)

  return abs(_phi(m + 1) - _phi(m))


def calculateHRV(bvpdata, sample_rate):
  bvpdata = list(map(lambda x: x['data'], bvpdata))
  nparr = np.asarray(bvpdata) #convert list to Phsio model with sanple rate 400

  f_data = hp.filtersignal(nparr,sample_rate=sample_rate,order=3, cutoff=[0.5, 4.0], filtertype='bandpass')
  working_data, measures = hp.process(f_data, sample_rate, calc_freq=True)
  measures['RR_avg'] = np.average(working_data['RR_list_cor'])
  hr = 60000/measures['RR_avg']
  measures['HR_avg'] = np.average(hr)
  measures['ApEn'] = ApEn(working_data['RR_list_cor'], 2, (0.2 * measures['sdnn']))
  measures['SampEn'] = sampen.sampen2(working_data['RR_list_cor'], 2, (0.2 * measures['sdnn']))

  SD1 = (1 / np.sqrt(2)) * measures['sdsd']  # measures the width of poincare cloud https://github.com/pickus91/HRV/blob/master/poincare.py
  SD2 = np.sqrt((2 * measures['sdnn'] ** 2) - (0.5 * measures['sdsd'] ** 2))  # measures the length of the poincare cloud
  measures['SD1'] = SD1
  measures['SD2'] = SD2
  # plot_object = hp.plotter(working_data, measures, show=False)
  # # plot_object.savefig('plot_1.jpg')  # saves the plot as JPEG image.
  # plot_object.show()  # displays plot
  return measures

def calculateSC(scdata, sample_rate):
  data = list(map(lambda x: x['data'], scdata))
  y = np.asarray(data) #convert list to Phsio model with sanple rate 400
  import cvxEDA
  yn = (y - y.mean()) / y.std()
  Fs = sample_rate
  [scr, p, scl, l, d, e, obj] = cvxEDA.cvxEDA(yn,  1./ Fs)
  scr_peaks = find_peaks(scr)

  sc_avg = np.average(y)

  scl_avg = np.average(scl)
  scl_slope = np.amax(scl) - np.amin(scl)

  scr_avg = np.average(scr)
  scr_max = np.amax(scr)

  scr_peaks_number = len(scr_peaks)

  result = {
    'sc_avg': sc_avg,
    'scl_avg': scl_avg,
    'scl_slope': scl_slope,
    'scr_avg': scr_avg,
    'scr_max': scr_max,
    'scr_peak': scr_peaks_number
  }

  # import pylab as pl
  # tm = pl.arange(1., len(y) + 1.) / Fs
  # #
  # # pl.plot(tm, yn)
  # pl.plot(tm, scr)
  # # pl.plot(tm, p)
  # #pl.plot(tm, scl)
  # pl.show()

  #processed_eda = neurokit.eda_process(nparr)
  #onsets, peaks, amplitudes, recoveries = neurokit.eda_scr(nparr)

  # plt.plot(scr)
  # plt.plot(scr_peaks[0],scr[scr_peaks[0]], "x")
  #
  # plt.show()
  return result

def _main():
  firebase = pyrebase.initialize_app(config)

  db = firebase.database()
  all_users = db.child("cho").child("users").get().val()

  users_data = []
  for user in all_users:
    data = get_user_data(all_users[user])
    users_data.append(data)

  ba_s_bvpdata = list(map(lambda x: x, users_data[0]['bvp']['ba_s_bvpdata']))
  bvp_time = ba_s_bvpdata[-1]['timeStamp'] - ba_s_bvpdata[0]['timeStamp']
  bvp_sample_rate = len(ba_s_bvpdata) / bvp_time

  ba_s_gsrdata = list(map(lambda x: x, users_data[0]['gsr']['ba_s_gsrdata']))
  gsr_time = ba_s_gsrdata[-1]['timeStamp'] - ba_s_gsrdata[0]['timeStamp']
  gsr_sample_rate = len(ba_s_gsrdata)/gsr_time

  ppg_features = calculateHRV(ba_s_bvpdata, bvp_sample_rate)
  eda_features = calculateSC(ba_s_gsrdata, gsr_sample_rate)

  x=1

  return users_data

def get_user_data(user):
  panelist={}
  panelist['BA-SEnd']=user['BA-SEnd']
  panelist['BA-SStart']=user['BA-SStart']
  panelist['MIS-SEnd']=user['MIS-SEnd']
  panelist['MIS-SStart']=user['MIS-SStart']
  panelist['MOS-SEnd']=user['MOS-SEnd']
  panelist['MOS-SStart']=user['MOS-SStart']
  panelist['SES-SStart']=user['SES-SStart']
  panelist['SES-SEnd']= 1554307848588

  ba_s_bvpdata = []
  mis_s_bvpdata = []
  mos_s_bvpdata = []
  ses_s_bvpdata = []

  for (k, v) in user['bvpData'].items():
    if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
      ba_s_bvpdata.append(v)
    if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MIS-SEnd'] / 1000):
      mis_s_bvpdata.append(v)
    if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MOS-SEnd'] / 1000):
      mos_s_bvpdata.append(v)
    if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(panelist['SES-SEnd'] / 1000):
      mos_s_bvpdata.append(v)

  ba_s_gsrdata = []
  mis_s_gsrdata = []
  mos_s_gsrdata = []
  ses_s_gsrdata = []
  for (k, v) in user['gsrData'].items():
    if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
      ba_s_gsrdata.append(v)
    if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MIS-SEnd'] / 1000):
      mis_s_gsrdata.append(v)
    if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MOS-SEnd'] / 1000):
      mos_s_gsrdata.append(v)
    if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(panelist['SES-SEnd'] / 1000):
      ses_s_gsrdata.append(v)

  ba_s_ibidata = []
  mis_s_ibidata = []
  mos_s_ibidata = []
  ses_s_ibidata = []

  for (k, v) in user['ibiData'].items():
    if v['timeStamp'] > float(panelist['BA-SStart'] / 1000) and v['timeStamp'] < float(panelist['BA-SEnd'] / 1000):
      ba_s_ibidata.append(v)
    if v['timeStamp'] > float(panelist['MIS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MIS-SEnd'] / 1000):
      mis_s_ibidata.append(v)
    if v['timeStamp'] > float(panelist['MOS-SStart'] / 1000) and v['timeStamp'] < float(panelist['MOS-SEnd'] / 1000):
      mos_s_ibidata.append(v)
    if v['timeStamp'] > float(panelist['SES-SStart'] / 1000) and v['timeStamp'] < float(panelist['SES-SEnd'] / 1000):
      ses_s_ibidata.append(v)


  data = {
    'bvp': {
      'ba_s_bvpdata': ba_s_bvpdata,
      'mis_s_bvpdata': mis_s_bvpdata,
      'mos_s_bvpdata': mos_s_bvpdata,
      'ses_s_bvpdata': ses_s_bvpdata
    },
    'gsr': {
      'ba_s_gsrdata': ba_s_gsrdata,
      'mis_s_gsrdata': mis_s_gsrdata,
      'mos_s_gsrdata': mos_s_gsrdata,
      'ses_s_gsrdata': ses_s_gsrdata
    },
    'ibi': {
      'ba_s_ibidata': ba_s_ibidata,
      'mis_s_ibidata': mis_s_ibidata,
      'mos_s_ibidata': mos_s_ibidata,
      'ses_s_ibidata': ses_s_ibidata
    },

  }
  return data


_main()