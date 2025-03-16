# Importing all necessary Libraies

import requests
from selenium import webdriver
import folium
import datetime
import time


def locationCoordinates():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        loc = data['loc'].split(',')
        lat, long = float(loc[0]), float(loc[1])
        city = data.get('city', 'Unknown')
        state = data.get('region', 'Unknown')
        return lat, long, city, state

    except:
        # Displaying ther error message
        print("Error in getting location coordinates")
        # closing the program
        exit()
        return False


print(locationCoordinates())
