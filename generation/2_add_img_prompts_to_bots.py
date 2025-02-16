import random, requests, time, json5, random, pytz
from datetime import datetime
from pymongo import MongoClient
from datetime import datetime

def getCurrentLocalTime():
    return datetime.now(pytz.timezone("Etc/GMT-5")).strftime("%H:%M:%S")


# generate 1-6 pics for profile & s3
# create simple api
# create pwa 
# upload pwa to play market
# find good marketing angles
# create ads
# run ads
# collect data & improve

def generate(messages):
    r = None
    while r is None:
        try:
            start = time.time()
            r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={"messages": messages}, timeout=55) # , timeout=55
            print(f"Took {time.time() - start}")
        except Exception as e:
            print(e)
    return r.json()['choices'][0]['message']['content']


def generateImgPrompts(bot):
    if not 'prompts' in bot:
        print('no prompts, cap\ngenerating new ones')
        prompt = """
        
        """


client = MongoClient('mongodb://localhost:27017/')

db = client['app']

users = db['users']
bots = db['bots']

bot = bots.find_one()

if bot:
    generateImgPrompts(bot)
else:
    print("No bots found.")
    
            
    