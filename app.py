import numpy as np
from flask import Flask, request, render_template
import pickle
import datetime
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="myGeocoder")

app = Flask(__name__)
model = pickle.load(open("NYC_Fare.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    temp_array = list()
    passenger = int(request.form["passenger"])
    #     date = (request.form["date"])
    date = datetime.datetime.strptime(request.form["date"], '%Y-%m-%dT%H:%M')
    print(date)
    hour = date.strftime("%H")

    # minute = date.strftime("%M")
    # print(datetime.time(6, 30, 00))
    # x = datetime.time(int(hour), int(minute), 00)
    weekday = date.strftime("%w")
    # print(weekday)
    pickup = str(request.form["pickup"])
    pickup = geolocator.geocode(pickup)
    dropoff = str(request.form["dropoff"])
    dropoff = geolocator.geocode(dropoff)

    def distance(lat1, lon1, lat2, lon2):
        p = 0.017453292519943295  # Pi/180
        a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (
                1 - np.cos((lon2 - lon1) * p)) / 2
        return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

    distance_in_miles = distance(pickup.latitude, pickup.longitude, dropoff.latitude, dropoff.longitude)

    distance = np.abs(pickup.latitude - dropoff.latitude) + np.abs(pickup.longitude - dropoff.longitude)
    # if (x >= datetime.time(6,30,00) and x < datetime.time(18,30,00)):
    #     day = 1
    # else : day = 0
    # if (x >= datetime.time(18,30,00) and x < datetime.time(22,30,00)):
    #     night = 1
    # else :
    #     night = 0
    # if (x >= datetime.time(22,30,00) or x < datetime.time(6,30,00)):
    #     late_night = 1
    # else :
    #     late_night = 0
    temp_array = temp_array + [passenger, weekday, hour, distance_in_miles, distance]

    data = np.array([temp_array])

    prediction = model.predict(data)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Your Cab Fare should be ${}'.format(output))
    # return render_template('index.html', prediction_text='day night vector {}'.format(distance))


if __name__ == '__main__':
    app.run(debug=True)
