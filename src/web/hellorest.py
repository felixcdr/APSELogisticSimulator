from flask import Flask, jsonify, request 

app = Flask(__name__) 
@app.route('/hello', methods=['GET']) 
def hello(): 
  response = {'message': 'Hello, world!'} 
  return jsonify(response), 200

app.run(host='0.0.0.0', port=8181)
