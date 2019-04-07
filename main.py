import pyrebase

config = {
  "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
  "authDomain": "synapsys-388ba.firebaseapp.com",
  "databaseURL": "https://synapsys-388ba.firebaseio.com/",
  "storageBucket": "synapsys-388ba.appspot.com"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
all_users = db.child("cho").child("users").get().val()
panelist={}

for user in all_users:
    print(user)
    panelist['BA-SEnd']=all_users[user]['BA-SEnd']
    panelist['BA-SStart']=all_users[user]['BA-SStart']
    panelist['MIS-SEnd']=all_users[user]['MIS-SEnd']
    panelist['MIS-SStart']=all_users[user]['MIS-SStart']
    panelist['MOS-SEnd']=all_users[user]['MOS-SEnd']
    panelist['MOS-SStart']=all_users[user]['MOS-SStart']
    panelist['SES-SStart']=all_users[user]['SES-SStart']
    panelist['SES-SEnd']=1554307848588
    m = list(filter(lambda x: all_users[user]['bvpData'][x]['timeStamp'] > float(panelist['BA-SStart']/1000) and all_users[user]['bvpData'][x]['timeStamp']<float(panelist['BA-SEnd']/1000), all_users[user]['bvpData']))

x=5