import pandas as pd
from flask import Flask, request, send_from_directory
import pickle
import os

model = pickle.load(open("video_games_category.pkl", "rb"))
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def verifier():
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pred = model.predict(df_raw)
        df_raw["predictions"] = pred

        return df_raw.to_json(orient="records")
    else:
        return "hello world"


if __name__ == '__main__':
    app.run(port="5500")