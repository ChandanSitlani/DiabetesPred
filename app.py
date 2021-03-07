from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response

import os

import pandas

from catboost import CatBoostClassifier

d=pandas.read_csv("diabetes_data_upload (1).csv")
print(len(d.iloc[0]))
def hello_world(request):
    print('Incoming request')
    return Response('Hello')

model=CatBoostClassifier(cat_features=list(d.columns))
print(list(d.columns))
model.load_model("Diabetes (1).cbm")
def predict(request):
    
    
    data=request.json_body
    inputs=[
        data["age"],
        data["gender"],
        data["polyuria"],
        data["polydipsia"],
        data["sudden_weight_loss"],
        data["weakness"],
        data["polyphagia"],
        data["genital_thrush"],
        data["visual_blurring"],
        data["itching"],
        data["irritability"],
        data["delayed_healing"],
        data["partial_paresis"],
        data["muscle_stiffness"],
        data["Alopecia"],
        data["obesity"],
        "Positive"
        ]
    print(inputs)
    print(request.json_body["age"])
    print(len(inputs))
    d.iloc[0]=inputs
    a=model.predict(d.iloc[0:2].drop(["class"],axis=1))[0]
    print(model.predict(d.iloc[307:309].drop(["class"],axis=1)))
    
    return Response(a);

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 2000))
    
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        config.add_route('predict', '/predict')
        config.add_view(predict, route_name='predict')
        app = config.make_wsgi_app()
    serve(app, host='0.0.0.0', port=port)
