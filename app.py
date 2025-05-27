import pandas as pd
import json
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from tensorflow import keras
sns.set_style("whitegrid") 
plt.style.use("fivethirtyeight") 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
warnings.filterwarnings("ignore") 
from datetime import date 
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import plotly
import plotly.utils
import json
import plotly.graph_objs as go


app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    file = FileField('file')
    submit = SubmitField('Upload File')

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        
        if file and file.filename.endswith('.csv'):
            new_filename = 'nickel.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            return render_template('index.html', tables=[df.to_html(classes='table table-striped')], titles=df.columns.values)
    return render_template('index.html')
    
@app.route('/predic.html/')
def predic_lstm():
    # Load and preprocess data
    stock_data = pd.read_csv('static/files/nickel.csv', encoding="UTF-8")
    
    def clean_convert(data):
        data_clean = data.replace(",", "")  # Menghapus koma tanpa mengganti dengan titik
        return float(data_clean)
    
    # Preprocessing
    stock_data['Price'] = stock_data['Price'].apply(clean_convert).astype('float64')
    stock_data['Open'] = stock_data['Open'].apply(clean_convert).astype('float64')
    stock_data['High'] = stock_data['High'].apply(clean_convert).astype('float64')
    stock_data['Low'] = stock_data['Low'].apply(clean_convert).astype('float64')
    
    stock_data = stock_data.drop(['Vol.', 'Change %'], axis=1)
    
    # perbaikan format date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
    stock_data = stock_data.sort_values('Date', ascending=True)
    stock_data.set_index('Date', inplace=True)
    
    stock_data = stock_data[['Price', 'High', 'Low', 'Open']]
    MMS = MinMaxScaler()
    stock_data[stock_data.columns] = MMS.fit_transform(stock_data)
    
    # Split data
    training_size = round(len(stock_data) * 0.80)
    train_data = stock_data[:training_size]
    test_data = stock_data[training_size:]
    
    def create_sequence(dataset):
        sequences = []
        labels = []
        start_idx = 0
        for stop_idx in range(50, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))
    
    X_train, y_train = create_sequence(train_data)
    X_test, y_test = create_sequence(test_data)
    
    # Load model and make predictions
    regressor = load_model('regressorlstm.h5')
    test_predicted = regressor.predict(X_test)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    
    # Create charts data
    merge_data = pd.concat([
        stock_data.iloc[-(len(test_data)-50):].copy(),
        pd.DataFrame(
            test_inverse_predicted,
            columns=['high_predicted','low_predicted','open_predicted','close_predicted'],
            index=stock_data.iloc[-(len(test_data)-50):].index
        )
    ], axis=1)
    
    merge_data[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        merge_data[['Price', 'High', 'Low', 'Open']]
    )
    
    # Gunakan tanggal terakhir dari data asli
    last_date = merge_data.index[-1]
    
    # Create future predictions
    new_dates = pd.DataFrame(
        columns=merge_data.columns,
        index=pd.date_range(start=last_date, periods=11, freq='D')
    )
    merge_data_2 = pd.concat([merge_data, new_dates])
    
    upcoming_prediction = pd.DataFrame(
        columns=['Price', 'High', 'Low', 'Open'],
        index=merge_data_2.index
    )
    upcoming_prediction.index = pd.to_datetime(upcoming_prediction.index)
    
    curr_seq = X_test[-1:]
    for i in range(-10, 0):
        up_pred = regressor.predict(curr_seq)
        upcoming_prediction.iloc[i] = up_pred
        curr_seq = np.append(curr_seq[0][1:], up_pred, axis=0)
        curr_seq = curr_seq.reshape(X_test[-1:].shape)
    
    upcoming_prediction[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        upcoming_prediction[['Price', 'High', 'Low', 'Open']]
    )
    
    # Create Plotly figures dengan data yang sudah diproses
    trace1 = go.Scatter(
        x=merge_data.index,
        y=merge_data['High'],
        name='Actual High',
        line=dict(color='blue')
    )
    
    trace2 = go.Scatter(
        x=merge_data.index,
        y=merge_data['high_predicted'],
        name='Predicted High',
        line=dict(color='cyan')
    )
    
    # Menggunakan tanggal yang sesuai untuk prediksi masa depan
    future_start_date = last_date
    trace3 = go.Scatter(
        x=merge_data_2.loc[future_start_date:].index,
        y=merge_data_2.loc[future_start_date:, 'High'],
        name='Current High Price',
        line=dict(color='cyan')
    )
    
    trace4 = go.Scatter(
        x=upcoming_prediction.loc[future_start_date:].index,
        y=upcoming_prediction.loc[future_start_date:, 'High'],
        name='Future High Price Prediction',
        line=dict(color='red')
    )
    
    # Create layouts
    layout1 = go.Layout(
        title='Actual dan Prediksi Untuk Harga Tertinggi (LSTM)',
        xaxis=dict(
            title='Tanggal',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga Nickel USD'),
        hovermode='x unified'
    )
    
    layout2 = go.Layout(
        title='Kemungkinan Prediksi Harga Tertinggi (High) (LSTM)',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga (USD)'),
        hovermode='x unified'
    )
    
    # Calculate metrics
    mse = mean_squared_error(y_test, test_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predicted)
    tertinggi = upcoming_prediction['High'].max()
    mae = mean_absolute_error(y_test, test_predicted)
    
    # Convert figures to JSON
    fig1 = go.Figure(data=[trace1, trace2], layout=layout1)
    fig2 = go.Figure(data=[trace3, trace4], layout=layout2)
    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        "predic.html",
        historical_chart=graphJSON1,
        prediction_chart=graphJSON2,
        mse_lstm=mse,
        rmse_lstm=rmse,
        r2_lstm=r2,
        high_lstm=tertinggi,
        test_lstm=mae
    )
    
@app.route("/bilstm.html")   
def predic_bilstm():
    stock_data = pd.read_csv('static/files/nickel.csv', encoding="UTF-8")
    
    def clean_convert(data):
        data_clean = data.replace(",", "")  # Menghapus koma tanpa mengganti dengan titik
        return float(data_clean)
    
    # Preprocessing
    stock_data['Price'] = stock_data['Price'].apply(clean_convert).astype('float64')
    stock_data['Open'] = stock_data['Open'].apply(clean_convert).astype('float64')
    stock_data['High'] = stock_data['High'].apply(clean_convert).astype('float64')
    stock_data['Low'] = stock_data['Low'].apply(clean_convert).astype('float64')
    
    stock_data = stock_data.drop(['Vol.', 'Change %'], axis=1)
    
    # Perbaikan format tanggal
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
    stock_data = stock_data.sort_values('Date', ascending=True)
    stock_data.set_index('Date', inplace=True)
    
    stock_data = stock_data[['Price', 'High', 'Low', 'Open']]
    MMS = MinMaxScaler()
    stock_data[stock_data.columns] = MMS.fit_transform(stock_data)
    
    training_size = round(len(stock_data) * 0.80)    
    train_data = stock_data[:training_size] #data train
    test_data  = stock_data[training_size:] #data test

    def create_sequence(dataset):
        sequences = []
        labels = []
        start_idx = 0
        for stop_idx in range(50, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))
    
    X_train, y_train = create_sequence(train_data)
    X_test, y_test = create_sequence(test_data)
    
    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    #model BiLSTM
    regressor = load_model('regressorbilstm.h5')
    test_predicted = regressor.predict(X_test)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    

    # Create charts data
    merge_data = pd.concat([
        stock_data.iloc[-(len(test_data)-50):].copy(),
        pd.DataFrame(
            test_inverse_predicted,
            columns=['high_predicted','low_predicted','open_predicted','close_predicted'],
            index=stock_data.iloc[-(len(test_data)-50):].index
        )
    ], axis=1)
    
    merge_data[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        merge_data[['Price', 'High', 'Low', 'Open']]
    )
    
    # Gunakan tanggal terakhir dari data asli
    last_date = merge_data.index[-1]
    
    # Create future predictions
    new_dates = pd.DataFrame(
        columns=merge_data.columns,
        index=pd.date_range(start=last_date, periods=11, freq='D'))
    merge_data_2 = pd.concat([merge_data, new_dates])
    
    upcoming_prediction = pd.DataFrame(
        columns=['Price', 'High', 'Low', 'Open'],
        index=merge_data_2.index
    )
    upcoming_prediction.index = pd.to_datetime(upcoming_prediction.index)
    
    curr_seq = X_test[-1:]

    curr_seq = X_test[-1:]
    for i in range(-10, 0):
        up_pred = regressor.predict(curr_seq)
        upcoming_prediction.iloc[i] = up_pred
        curr_seq = np.append(curr_seq[0][1:], up_pred, axis=0)
        curr_seq = curr_seq.reshape(X_test[-1:].shape)
    
    upcoming_prediction[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        upcoming_prediction[['Price', 'High', 'Low', 'Open']]
    )
    # Create Plotly figures dengan data yang sudah diproses
    trace1 = go.Scatter(
        x=merge_data.index,
        y=merge_data['High'],
        name='Actual High',
        line=dict(color='blue')
    )
    
    trace2 = go.Scatter(
        x=merge_data.index,
        y=merge_data['high_predicted'],
        name='Predicted High',
        line=dict(color='cyan')
    )
    
    # Menggunakan tanggal yang sesuai untuk prediksi masa depan
    future_start_date = last_date
    trace3 = go.Scatter(
        x=merge_data_2.loc[future_start_date:].index,
        y=merge_data_2.loc[future_start_date:, 'High'],
        name='Current High Price',
        line=dict(color='cyan')
    )
    
    trace4 = go.Scatter(
        x=upcoming_prediction.loc[future_start_date:].index,
        y=upcoming_prediction.loc[future_start_date:, 'High'],
        name='Future High Price Prediction',
        line=dict(color='red')
    )
    
    # Create layouts
    layout1 = go.Layout(
        title='Actual dan Prediksi Untuk Harga Tertinggi (BiLSTM)',
        xaxis=dict(
            title='Tanggal',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga Nickel USD'),
        hovermode='x unified'
    )
    
    layout2 = go.Layout(
        title='Kemungkinan Prediksi Harga Tertinggi (High) (BiLSTM)',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga (USD)'),
        hovermode='x unified'
    )
    
    # Calculate metrics
    mse = mean_squared_error(y_test, test_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predicted)
    tertinggi = upcoming_prediction['High'].max()
    mae = mean_absolute_error(y_test, test_predicted)
    
    # Convert figures to JSON
    fig1 = go.Figure(data=[trace1, trace2], layout=layout1)
    fig2 = go.Figure(data=[trace3, trace4], layout=layout2)
    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        "bilstm.html",
        historical_chart=graphJSON1,
        prediction_chart=graphJSON2,
        mse_bilstm=mse,
        rmse_bilstm=rmse,
        r2_bilstm=r2,
        high_bilstm=tertinggi,
        test_bilstm=mae
    )
@app.route("/bigru.html")
def predic_bigru():
    stock_data = pd.read_csv('static/files/nickel.csv', encoding="UTF-8")
    
    def clean_convert(data):
        data_clean = data.replace(",", "")  # Menghapus koma tanpa mengganti dengan titik
        return float(data_clean)
    
    # Preprocessing
    stock_data['Price'] = stock_data['Price'].apply(clean_convert).astype('float64')
    stock_data['Open'] = stock_data['Open'].apply(clean_convert).astype('float64')
    stock_data['High'] = stock_data['High'].apply(clean_convert).astype('float64')
    stock_data['Low'] = stock_data['Low'].apply(clean_convert).astype('float64')
    
    stock_data = stock_data.drop(['Vol.', 'Change %'], axis=1)
    
    # Perbaikan format tanggal
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
    stock_data = stock_data.sort_values('Date', ascending=True)
    stock_data.set_index('Date', inplace=True)
    
    stock_data = stock_data[['Price', 'High', 'Low', 'Open']]
    MMS = MinMaxScaler()
    stock_data[stock_data.columns] = MMS.fit_transform(stock_data)
    
    training_size = round(len(stock_data) * 0.80)    
    train_data = stock_data[:training_size] #data train
    test_data  = stock_data[training_size:] #data test

    def create_sequence(dataset):
        sequences = []
        labels = []
        start_idx = 0
        for stop_idx in range(50, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))
    
    X_train, y_train = create_sequence(train_data)
    X_test, y_test = create_sequence(test_data)
    
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #model BiGRU
    regressor = load_model('regressorbigru.h5')
    test_predicted = regressor.predict(X_test)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    

    # Create charts data
    merge_data = pd.concat([
        stock_data.iloc[-(len(test_data)-50):].copy(),
        pd.DataFrame(
            test_inverse_predicted,
            columns=['high_predicted','low_predicted','open_predicted','close_predicted'],
            index=stock_data.iloc[-(len(test_data)-50):].index
        )
    ], axis=1)
    
    merge_data[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        merge_data[['Price', 'High', 'Low', 'Open']]
    )
    
    # Gunakan tanggal terakhir dari data asli
    last_date = merge_data.index[-1]
    
    # Create future predictions
    new_dates = pd.DataFrame(
        columns=merge_data.columns,
        index=pd.date_range(start=last_date, periods=11, freq='D'))
    merge_data_2 = pd.concat([merge_data, new_dates])
    
    upcoming_prediction = pd.DataFrame(
        columns=['Price', 'High', 'Low', 'Open'],
        index=merge_data_2.index
    )
    upcoming_prediction.index = pd.to_datetime(upcoming_prediction.index)
    
    curr_seq = X_test[-1:]
    
    for i in range(-10, 0):
        up_pred = regressor.predict(curr_seq)
        upcoming_prediction.iloc[i] = up_pred
        curr_seq = np.append(curr_seq[0][1:], up_pred, axis=0)
        curr_seq = curr_seq.reshape(X_test[-1:].shape)
    
    upcoming_prediction[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        upcoming_prediction[['Price', 'High', 'Low', 'Open']]
    )
    # Create Plotly figures dengan data yang sudah diproses
    trace1 = go.Scatter(
        x=merge_data.index,
        y=merge_data['High'],
        name='Actual High',
        line=dict(color='blue')
    )
    
    trace2 = go.Scatter(
        x=merge_data.index,
        y=merge_data['high_predicted'],
        name='Predicted High',
        line=dict(color='cyan')
    )
    
    # Menggunakan tanggal yang sesuai untuk prediksi masa depan
    future_start_date = last_date
    trace3 = go.Scatter(
        x=merge_data_2.loc[future_start_date:].index,
        y=merge_data_2.loc[future_start_date:, 'High'],
        name='Current High Price',
        line=dict(color='cyan')
    )
    
    trace4 = go.Scatter(
        x=upcoming_prediction.loc[future_start_date:].index,
        y=upcoming_prediction.loc[future_start_date:, 'High'],
        name='Future High Price Prediction',
        line=dict(color='red')
    )
    
    # Create layouts
    layout1 = go.Layout(
        title='Actual dan Prediksi Untuk Harga Tertinggi (BiGRU)',
        xaxis=dict(
            title='Tanggal',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga Nickel USD'),
        hovermode='x unified'
    )
    
    layout2 = go.Layout(
        title='Kemungkinan Prediksi Harga Tertinggi (High) (BiGRU)',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga (USD)'),
        hovermode='x unified'
    )
    
    # Calculate metrics
    mse = mean_squared_error(y_test, test_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predicted)
    tertinggi = upcoming_prediction['High'].max()
    mae = mean_absolute_error(y_test, test_predicted)
    
    # Convert figures to JSON
    fig1 = go.Figure(data=[trace1, trace2], layout=layout1)
    fig2 = go.Figure(data=[trace3, trace4], layout=layout2)
    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        "bigru.html",
        historical_chart=graphJSON1,
        prediction_chart=graphJSON2,
        mse_bigru=mse,
        rmse_bigru=rmse,
        r2_bigru=r2,
        high_bigru=tertinggi,
        test_bigru=mae
    )
@app.route("/gru.html/")
def predic_Gru():
    stock_data = pd.read_csv('static/files/nickel.csv', encoding="UTF-8")
    def clean_convert(data):
        data_clean = data.replace(",", "")  # Menghapus koma tanpa mengganti dengan titik
        return float(data_clean)
    
    # Preprocessing
    stock_data['Price'] = stock_data['Price'].apply(clean_convert).astype('float64')
    stock_data['Open'] = stock_data['Open'].apply(clean_convert).astype('float64')
    stock_data['High'] = stock_data['High'].apply(clean_convert).astype('float64')
    stock_data['Low'] = stock_data['Low'].apply(clean_convert).astype('float64')
    
    stock_data = stock_data.drop(['Vol.', 'Change %'], axis=1)
    
    # Perbaikan format tanggal
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
    stock_data = stock_data.sort_values('Date', ascending=True)
    stock_data.set_index('Date', inplace=True)
    
    stock_data = stock_data[['Price', 'High', 'Low', 'Open']]
    MMS = MinMaxScaler()
    stock_data[stock_data.columns] = MMS.fit_transform(stock_data)
    
    training_size = round(len(stock_data) * 0.80)    
    train_data = stock_data[:training_size] #data train
    test_data  = stock_data[training_size:] #data test
   
    def create_sequence(dataset):
        sequences = []
        labels = []
        start_idx = 0
        for stop_idx in range(50, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))
    
    X_train, y_train = create_sequence(train_data)
    X_test, y_test = create_sequence(test_data)
    
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #model GRU
    regressor = load_model('regressorgru1.h5')
    test_predicted = regressor.predict(X_test)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    

    # Create charts data
    merge_data = pd.concat([
        stock_data.iloc[-(len(test_data)-50):].copy(),
        pd.DataFrame(
            test_inverse_predicted,
            columns=['high_predicted','low_predicted','open_predicted','close_predicted'],
            index=stock_data.iloc[-(len(test_data)-50):].index
        )
    ], axis=1)
    
    merge_data[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        merge_data[['Price', 'High', 'Low', 'Open']]
    )
    # Gunakan tanggal terakhir dari data asli
    last_date = merge_data.index[-1]
    
    # Create future predictions
    new_dates = pd.DataFrame(
        columns=merge_data.columns,
        index=pd.date_range(start=last_date, periods=11, freq='D'))
    merge_data_2 = pd.concat([merge_data, new_dates])
    
    upcoming_prediction = pd.DataFrame(
        columns=['Price', 'High', 'Low', 'Open'],
        index=merge_data_2.index
    )
    upcoming_prediction.index = pd.to_datetime(upcoming_prediction.index)
    
    curr_seq = X_test[-1:]

    for i in range(-10,0):
        up_pred = regressor.predict(curr_seq)
        upcoming_prediction.iloc[i] = up_pred
        curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
        curr_seq = curr_seq.reshape(X_test[-1:].shape)
        
    upcoming_prediction[['Price', 'High', 'Low', 'Open']] = MMS.inverse_transform(
        upcoming_prediction[['Price', 'High', 'Low', 'Open']]
    )
    # Create Plotly figures dengan data yang sudah diproses
    trace1 = go.Scatter(
        x=merge_data.index,
        y=merge_data['High'],
        name='Actual High',
        line=dict(color='blue')
    )
    
    trace2 = go.Scatter(
        x=merge_data.index,
        y=merge_data['high_predicted'],
        name='Predicted High',
        line=dict(color='cyan')
    )
    
    # Menggunakan tanggal yang sesuai untuk prediksi masa depan
    future_start_date = last_date
    trace3 = go.Scatter(
        x=merge_data_2.loc[future_start_date:].index,
        y=merge_data_2.loc[future_start_date:, 'High'],
        name='Current High Price',
        line=dict(color='cyan')
    )
    
    trace4 = go.Scatter(
        x=upcoming_prediction.loc[future_start_date:].index,
        y=upcoming_prediction.loc[future_start_date:, 'High'],
        name='Future High Price Prediction',
        line=dict(color='red')
    )
    
    # Create layouts
    layout1 = go.Layout(
        title='Actual dan Prediksi Untuk Harga Tertinggi (GRU)',
        xaxis=dict(
            title='Tanggal',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga Nickel USD'),
        hovermode='x unified'
    )
    
    layout2 = go.Layout(
        title='Kemungkinan Prediksi Harga Tertinggi (High) (GRU)',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            )
        ),
        yaxis=dict(title='Harga (USD)'),
        hovermode='x unified'
    )
    
    # Calculate metrics
    mse = mean_squared_error(y_test, test_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predicted)
    tertinggi = upcoming_prediction['High'].max()
    mae = mean_absolute_error(y_test, test_predicted)
    
    # Convert figures to JSON
    fig1 = go.Figure(data=[trace1, trace2], layout=layout1)
    fig2 = go.Figure(data=[trace3, trace4], layout=layout2)
    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        "gru.html",
        historical_chart=graphJSON1,
        prediction_chart=graphJSON2,
        mse_gru=mse,
        rmse_gru=rmse,
        r2_gru=r2,
        high_gru=tertinggi,
        test_gru=mae
    )

if __name__ == '__main__':
    app.run(debug=True)