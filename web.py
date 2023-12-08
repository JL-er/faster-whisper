# -*- coding: UTF-8 -*-
from flask_cors import CORS
from flask import Flask, request, Response, render_template, jsonify
import json

# 初始化flaskAPP
#from RWKV.v2.interfence import on_message
from inf import whisper
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


@app.route('/api/voice', methods=['POST'])
def button():

    mp3_path =request.json.get('message')
    #reply = "hhhhhhh"
    reply = whisper(mp3_path)
    return jsonify({'reply': reply})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5600, use_reloader=False)