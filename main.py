import pyrebase

config = {
  "apiKey": "AIzaSyB_amArWXiqAOMdpHsTQVkr62h2jN-YK9M",
  "authDomain": "synapsys-388ba.firebaseapp.com",
  "databaseURL": "https://synapsys-388ba.firebaseio.com/",
  "storageBucket": "synapsys-388ba.appspot.com"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
all_users = db.child("cho").child("users").get()
x=5