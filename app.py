from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    response = "Response for the query: " + query
    return jsonify({'query': query, 'response': response})

if __name__ == '__main__':
    app.run(debug=True)