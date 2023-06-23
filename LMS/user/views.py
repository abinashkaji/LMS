import datetime
from django.conf import settings
from django.shortcuts import render
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import os

def index(request):
    # Assuming you have two sets of dates and corresponding values
    path=os.path.join(settings.BASE_DIR,'user/tesla.csv')
    df=pd.read_csv(path)
    dates = df.iloc[:,0].values[:-6]
    values=df.Close.values[:-6]
    name=''
    print(df.head())

    trace1 = go.Scatter(x=dates, y=values, mode='lines',  line=dict(color='blue'), name='Close')
    dates = df.iloc[:,0].values[-7:]
    values=df.Close.values[-7:]
    trace2 = go.Scatter(x=dates, y=values, mode='lines', line=dict(dash='dash',color='rgb(100,100,0)'), name='Predicted_close')    
    layout = go.Layout(title=f'{name} Stock', xaxis=dict(title='Date'), yaxis=dict(title='price $'))
    trace=[trace1,trace2]
    fig = go.Figure(data=trace, layout=layout)
    fig.add_shape(type="line", x0=datetime.date.today(), x1=datetime.date.today(), y0=0, y1=max(values)+50, line=dict(color="red", width=2))

    div = pyo.plot(fig, output_type='div',config= {'displaylogo': False}, include_plotlyjs=False)
    return render(request,'user/index.html',{'div':div})    