import pyrebase

config = {
  "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
  "authDomain": "synapsys-388ba.firebaseapp.com",
  "databaseURL": "https://synapsys-388ba.firebaseio.com/",
  "storageBucket": "synapsys-388ba.appspot.com"
}



def _main():
  firebase = pyrebase.initialize_app(config)

  db = firebase.database()
  all_users = db.child("cho").child("users").get().val()

  users_data = []
  for user in all_users:
    data = get_user_data(all_users[user])
    users_data.append(data)

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

  ba_s_bvpdata = list(filter(lambda x: user['bvpData'][x]['timeStamp'] > float(panelist['BA-SStart']/1000) and user['bvpData'][x]['timeStamp']<float(panelist['BA-SEnd']/1000), user['bvpData']))
  mis_s_bvpdata = list(filter(lambda x: user['bvpData'][x]['timeStamp'] > float(panelist['MIS-SStart']/1000) and user['bvpData'][x]['timeStamp']<float(panelist['MIS-SEnd']/1000), user['bvpData']))
  mos_s_bvpdata = list(filter(lambda x: user['bvpData'][x]['timeStamp'] > float(panelist['MOS-SStart']/1000) and user['bvpData'][x]['timeStamp']<float(panelist['MOS-SEnd']/1000), user['bvpData']))
  ses_s_bvpdata = list(filter(lambda x: user['bvpData'][x]['timeStamp'] > float(panelist['SES-SStart']/1000) and user['bvpData'][x]['timeStamp']<float(panelist['SES-SEnd']/1000), user['bvpData']))

  ba_s_gsrdata = list(filter(lambda x: user['gsrData'][x]['timeStamp'] > float(panelist['BA-SStart']/1000) and user['gsrData'][x]['timeStamp']<float(panelist['BA-SEnd']/1000), user['gsrData']))
  mis_s_gsrdata = list(filter(lambda x: user['gsrData'][x]['timeStamp'] > float(panelist['MIS-SStart']/1000) and user['gsrData'][x]['timeStamp']<float(panelist['MIS-SEnd']/1000), user['gsrData']))
  mos_s_gsrdata = list(filter(lambda x: user['gsrData'][x]['timeStamp'] > float(panelist['MOS-SStart']/1000) and user['gsrData'][x]['timeStamp']<float(panelist['MOS-SEnd']/1000), user['gsrData']))
  ses_s_gsrdata = list(filter(lambda x: user['gsrData'][x]['timeStamp'] > float(panelist['SES-SStart']/1000) and user['gsrData'][x]['timeStamp']<float(panelist['SES-SEnd']/1000), user['gsrData']))

  ba_s_ibidata = list(filter(lambda x: user['ibiData'][x]['timeStamp'] > float(panelist['BA-SStart']/1000) and user['ibiData'][x]['timeStamp']<float(panelist['BA-SEnd']/1000), user['ibiData']))
  mis_s_ibidata = list(filter(lambda x: user['ibiData'][x]['timeStamp'] > float(panelist['MIS-SStart']/1000) and user['ibiData'][x]['timeStamp']<float(panelist['MIS-SEnd']/1000), user['ibiData']))
  mos_s_ibidata = list(filter(lambda x: user['ibiData'][x]['timeStamp'] > float(panelist['MOS-SStart']/1000) and user['ibiData'][x]['timeStamp']<float(panelist['MOS-SEnd']/1000), user['ibiData']))
  ses_s_ibidata = list(filter(lambda x: user['ibiData'][x]['timeStamp'] > float(panelist['SES-SStart']/1000) and user['ibiData'][x]['timeStamp']<float(panelist['SES-SEnd']/1000), user['ibiData']))


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