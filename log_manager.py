APP_NAME="RAG Chat App"
from datetime import datetime


def print_log(msg):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print(f"{date_time} - {APP_NAME} : {msg}")

