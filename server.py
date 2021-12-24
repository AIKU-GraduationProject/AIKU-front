from re import template
from flask import Flask, render_template
from flask_cors import CORS
from flask import request
import json
# from ..model.model import infer
from merge_models import *
import pickle

# -- coding: utf-8 --
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

# load data
# with open('demo/pickle/code2cate2.pickle', 'rb') as fr:
#     code2cate = pickle.load(fr)
# with open('demo/pickle/code2name2.pickle', 'rb') as fr:
#     code2name = pickle.load(fr)

# 뒤에 resource path X, 브라우저 접근
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    data = str(request.json)
    print(data)
    result = infer(data)
    """
    json_data = {f"{idx} code" : value for idx, value in enumerate(result)}
    json_data.update({f"{idx} name" : code2name[value] for idx, value in enumerate(result)})
    json_data.update({f"{idx} category" : code2cate[value] for idx, value in enumerate(result)})
    json_data['context'] = data
    print(json_data)
    """
    json_data = {}
    send = json.dumps(json_data)
    return send

@app.route('/get_passage', methods=['POST'])
def get_relevant_docs():
    data = str(request.json)
    print(data)
    _, passages = query2passages(data)
    """
    json_data = {f"{idx} code" : value for idx, value in enumerate(result)}
    json_data.update({f"{idx} name" : code2name[value] for idx, value in enumerate(result)})
    json_data.update({f"{idx} category" : code2cate[value] for idx, value in enumerate(result)})
    json_data['context'] = data
    print(json_data)
    """
    json_data = {f"{idx+1} passage" : value for idx, value in enumerate(passages)}
    # json_data = {}
    send = json.dumps(json_data)
    return send

@app.route('/conversation', methods=['POST'])
def get_answer():
    data = request.json
    print(data)
    query = data['query']
    passage = data['passage']
    # query, passage 보냄
    answer = conversation(query, passage)
    """
    json_data = {f"{idx} code" : value for idx, value in enumerate(result)}
    json_data.update({f"{idx} name" : code2name[value] for idx, value in enumerate(result)})
    json_data.update({f"{idx} category" : code2cate[value] for idx, value in enumerate(result)})
    json_data['context'] = data
    print(json_data)
    """
    
    json_data = {'answer':answer}
    send = json.dumps(json_data)
    return send

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=False)