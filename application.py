import os

from flask import Flask, render_template
import logging

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'

# 디버깅을 위한 로그 설정
logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def home():
    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
