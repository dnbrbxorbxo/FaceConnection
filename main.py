from application import app

if __name__ == "__main__":
    app.debug=True
    app.secret_key = "temp key"
    app.run(debug=True, port=8000)