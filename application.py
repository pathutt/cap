import math
from flask import *
import os
import algorithms


app = Flask(__name__)
app.secret_key = os.urandom(12)


@app.route('/', methods=['GET', 'POST'])
def main():
    # Entry Point For Web App
    if request.method == "POST":
        cursor = algorithms.connect()
        cursor.execute("SELECT * FROM USERNAME")
        row = cursor.fetchone()
        while row:
            username = row[0]
            password = row[1]
            row = cursor.fetchone()
            if request.form['username'] == username and request.form['password'] == password:
                return home()
        wrong = 'Incorrect Username or Password'
        return render_template('main.html', incorrect=wrong)
    else:
        return render_template('main.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    price = None

    vis1_url = algorithms.create_kmeans_graph()
    vis2_url = algorithms.create_lin_reg_graph()
    vis3_url = algorithms.create_avg_vol_month_graph()

    if request.method == 'POST':
        price = request.form.get("price", None)

    def calculate_vol(price):
        try:
            price = float(price)
        except:
            return 'Please enter a number'

        if price <= 0:
            return 'Please enter number above 0'
        else:
            return math.floor(algorithms.volume_estimate(price))

    if price:
        vol = calculate_vol(price)
        return render_template('home.html', vis1=vis1_url, vis2=vis2_url, vis3=vis3_url, price=price, volume=vol)

    return render_template('home.html', vis1=vis1_url, vis2=vis2_url, vis3=vis3_url)



