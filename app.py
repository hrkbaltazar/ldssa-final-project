import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import datetime
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField, BooleanField,
    FloatField, TextField, IntegrityError, DateTimeField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


# curl -X POST http://localhost:5000/should_search/ -d '{"observation_id": "test", "Type": "Person search", "Part of a policing operation": "True",  "Latitude": 1.0, "Longitude": 1.0, "Gender": "Male", "Age range": "18-24", "Date" : "03/09/1991", "Officer-defined ethnicity": "Asian", "Legislation": "Misuse of Drugs Act 1971 (section 23)", "Object of search": "Controlled drugs", "station": "devon-and-cornwall"}' -H "Content-Type:application/json"

# curl -X POST https://capstone-henrique.herokuapp.com/should_search/ -d '{"observation_id": "test", "Type": "Person search", "Part of a policing operation": true,  "Latitude": 1.0, "Longitude": 1.0, "Gender": "Male", "Age range": "18-24", "Date" : "03/09/1991", "Officer-defined ethnicity": "Asian", "Legislation": "Misuse of Drugs Act 1971 (section 23)", "Object of search": "Controlled drugs", "station": "devon-and-cornwall"}' -H "Content-Type:application/json"


########################################
# Begin database stuff
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    predicted_outcome = BooleanField(null=True)
    outcome = BooleanField(null=True)
    date_received = DateTimeField(default=datetime.datetime.now)


    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json', 'r') as fh:
    columns = json.load(fh)


with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions


def check_request(request):

    if "observation_id" not in request:
        error = "Field `id` missing from request: {}".format(request)

        return False, error

    return True, ""


def check_valid_column(observation):

    valid_columns = {'observation_id', 'Type', 'Date', 'Part of a policing operation',
       'Latitude', 'Longitude', 'Gender', 'Age range',
       'Officer-defined ethnicity', 'Legislation',
       'Object of search', 'station'} #  'Removal of more than just outer clothing',

    keys = set(observation.keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error

    return True, ""

def predict_cluster(obs, coors):


    print("predicting cluster...")
    station_name = obs.station
    with open(f'clusters/km-{station_name}.pickle', 'rb') as fh:
        km = joblib.load(fh)
    cluster = km.predict([coors])
    print("cluster precited: " + str(cluster))
    
    return station_name + str(cluster[0])


def pre_process(df_original):
    
    df = df_original.copy()
    df.loc[(pd.isnull(df['Part of a policing operation'])), 'Part of a policing operation'] = False

    return df

# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():

    obs_dict = request.get_json()
    request_ok, error = check_request(obs_dict)

    if not request_ok:
        response = {'error': error}
        return jsonify(response)

    observation = obs_dict

    date = pd.to_datetime(observation['Date'][:19], infer_datetime_format=True)
    observation['hour'] = date.hour
    observation['month'] = date.month
    observation['day_of_week'] = date.day_name()

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)

    obs = pre_process(obs)

    if observation['Latitude'] != None:
        coors = [observation['Latitude'], observation['Longitude']]
        obs['cluster'] = obs.apply(lambda row: predict_cluster(row, coors), axis = 1)
    else:
        obs['cluster'] = None


    prediction_percent = pipeline.predict(obs)[0]
    predicted_outcome = True if prediction_percent > 0.228 else False
    response = {'outcome': predicted_outcome}

    _id = obs_dict['observation_id']

    p = Prediction(
        observation_id=_id,
        observation=request.data,
        predicted_outcome=predicted_outcome,
        outcome = None
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
        
    return jsonify(response)

#{"observation_id": "1", "Type": "Person search", "Part of a policing operation": "True",
#"Latitude": "1", "Longitude": "1", "Gender": "Male", "Age range": "18-24",
#"Date" : "03/09/1991", "Officer-defined ethnicity": "Asian", "Legislation":
#"Misuse of Drugs Act 1971 (section 23)", "Object of search": "Controlled drugs", "station": "devon-and-cornwall"}

@app.route('/search_result/', methods=['POST'])
def search_result():
    obs = request.get_json()

    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.outcome = bool(obs['outcome'])

        response = {"observation_id": obs['observation_id'] , 'outcome': bool(obs['outcome']), 'predicted_outcome': p.predicted_outcome}
        p.save()

        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})



if __name__ == "__main__":
    app.run()
