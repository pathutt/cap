import base64
import io
import math

import pyodbc
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def connect():
    server = 'c964.database.windows.net'
    database = 'c964'
    username = 'phutt'
    password = '@Llie0720'
    driver = '{ODBC Driver 17 for SQL Server}'
    cnxn = pyodbc.connect(
        'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    return cursor

def linear_reg():
    x = get_x_lin_reg()
    y = get_y_lin_reg()
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    return model


def get_x_lin_reg():
    cursor = connect()
    cursor.execute(
        "SELECT AveragePrice, Total_Volume from avocado where (region='Detroit' or region='GrandRapids') and type='conventional'")
    x = []
    row = cursor.fetchone()
    while row:
        x.append(float(row[0]))
        row = cursor.fetchone()
    return x


def get_y_lin_reg():
    cursor = connect()
    cursor.execute(
        "SELECT AveragePrice, Total_Volume from avocado where (region='Detroit' or region='GrandRapids') and type='conventional'")
    y = []
    row = cursor.fetchone()
    while row:
        y.append(float(row[1]))
        row = cursor.fetchone()
    return y


def create_lin_reg_graph():
    model = linear_reg()
    x = np.array(get_x_lin_reg()).reshape((-1, 1))
    y = np.array(get_y_lin_reg())
    plt.scatter(x, y, color='g')
    plt.plot(x, model.predict(x), color='k')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


def get_kmeans_values():
    vals = []
    cursor = connect()
    cursor.execute(
        "SELECT AveragePrice, Total_Volume from avocado where (region='Detroit' or region='GrandRapids') and type='conventional'")
    row = cursor.fetchone()
    while row:
        x = [float(row[0]), float(row[1])]
        vals.append(x)
        row = cursor.fetchone()
    np_vals = np.array(vals)
    return np_vals


def create_kmeans_graph():
    X = get_kmeans_values()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


def get_values_avg_vol():
    cursor = connect()
    cursor.execute("select floor(avg(Total_Volume)), Month(Date) from avocado where (region = 'Detroit' or region = "
                   "'GrandRapids') and type = 'conventional' group by Month(Date)")
    avg_vol = []
    row = cursor.fetchone()
    while row:
        avg_vol.append(float(row[0]))
        row = cursor.fetchone()
    return avg_vol


def create_avg_vol_month_graph():
    height = get_values_avg_vol()
    bars = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    y_pos = np.arange(len(bars))

    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


def volume_estimate(x):
    model = linear_reg()
    pred = model.intercept_ + model.coef_ * x
    return pred;


def kmeans_extimate(x):
    y = volume_estimate(x)
    new = get_kmeans_values()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(new)

    arr = np.array([x, y]).reshape(1, -1)

    return kmeans.fit(new).predict(arr)


