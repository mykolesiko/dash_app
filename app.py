import datetime

from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import base64
import os
import glob
import numpy as np
from src.model import ModelPredict
from src.transforms import val_transforms
from src.constants import NAMES
import torch
from flask import Flask


data_dir = "./data/"
model_dir = "./models/"
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = Dash(server = server, external_stylesheets=external_stylesheets)
#model_to_predict = None
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model_to_predict = ModelPredict(model_dir, data_dir, device, val_transforms)
    # image_file = glob.glob(os.path.join(data_dir, image_name))
    # image_name = np.random.choice(image_name)
model_to_predict.load_model("dogs_segmented_best_881.pth")
    

#@server.before_first_request
#def init():
#server = app.server
#    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#    model_to_predict = ModelPredict(model_dir, data_dir, device, val_transforms)
    # image_file = glob.glob(os.path.join(data_dir, image_name))
    # image_name = np.random.choice(image_name)
 #   model_to_predict.load_model("dogs_segmented_best_881.pth")
    


app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File with Dog'), 'and check'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div([
    	#html.Button('Check the dog', id='check_dog', n_clicks=0),
	html.Button('Random', id='next', n_clicks=0),
    	html.Div(id='check_result',
             children='result'),

    html.Div(id='output-image-upload'),

])
])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-image-upload', 'children'),
	      Output('check_result', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'),
	      Input('next', 'n_clicks'))
def update_output(list_of_contents, list_of_names, list_of_dates, n_clicks):
     changed_id = [p['prop_id'] for p in callback_context.triggered][0]
     if 'next' in changed_id:
            files = glob.glob(os.path.join("./data/", f"*/*.JPEG"))
            #print(files)
            image_name = np.random.choice(files)
            #prediction = make_predict(data_dir, model_dir, "best_resnet.pth", image_name[7:])
            prediction = model_to_predict.predict(image_name[7:])
            with open(image_name, 'rb') as image_file:
              encoded_string = base64.b64encode(image_file.read())
              result =  "data:image/jpg;base64," + encoded_string.decode('ascii')
              return html.Div([html.Img(src=result)]), "real " + NAMES[prediction] + " predicted " + model_to_predict.get_name(image_name[7:])


     if list_of_contents is not None:
         print("1")
         content = list_of_contents[0]
         data = content.encode("utf8").split(b";base64,")[1]
         print("2")
         with open(os.path.join("./data/", "temp.jpg"), "wb") as fp:
                 fp.write(base64.decodebytes(data))
         print("3")
         prediction = model_to_predict.predict("temp.jpg")
         print(prediction)
            
         children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]                                   
         return children , NAMES[prediction]
     return [], ""
    
   


if __name__ == '__main__':
    #IP  = os.popen('curl -s ifconfig.me').readline()
    app.run_server()