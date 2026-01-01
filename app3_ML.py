import os
import threading
import pandas as pd
import logging
from collections import deque

import dash
from dash.dependencies import Output, Input
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_chartjs import ChartJs 

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# --- 1. Global Setup & Data Buffers ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
data_lock = threading.Lock()

plant_types = ['Gas Plant', 'Wind Farm', 'Solar Farm', 'Hydroelectric Plant']

# Deque keeps only the last 60 data points (1 minute of data at 1s intervals)
# This prevents the "needle-thin" line issue by limiting the horizontal density.
data_store = {pt: deque(maxlen=60) for pt in plant_types}

# --- 2. Spark Streaming Logic ---
def start_spark_streaming():
    spark = SparkSession.builder.appName("EnergyAnalytics").master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1").getOrCreate()

    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("plant_type", StringType()),
        StructField("power_output", DoubleType()),
        StructField("demand", DoubleType())
    ])

    # Connect to Kafka
    df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')) \
        .option("subscribe", "energy_stream").load()

    # Parse JSON
    df_parsed = df.selectExpr("CAST(value AS STRING)") \
                  .select(from_json(col("value"), schema).alias("data")).select("data.*")

    def process_batch(batch_df, batch_id):
        pdf = batch_df.toPandas()
        if not pdf.empty:
            with data_lock:
                for pt in plant_types:
                    p_df = pdf[pdf['plant_type'] == pt]
                    if not p_df.empty:
                        data_store[pt].extend(p_df.to_dict('records'))

    df_parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()

# --- 3. Dash UI Components ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

def create_chart_card(title, chart_id):
    """Creates a fixed-size card containing a Chart.js canvas."""
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className="text-center m-0", style={'fontWeight': 'bold'})),
        dbc.CardBody([
            ChartJs(
                id=chart_id, 
                type='line', 
                options={
                    'responsive': True,
                    'maintainAspectRatio': False, # Allows us to control height via CSS
                    'animation': False,           # Disabling animation prevents 'shaking'
                    'scales': {
                        'y': {
                            'min': 0, 
                            'max': 160,           # Static Y-axis stops vertical jumping
                            'title': {'display': True, 'text': 'Megawatts (MW)'}
                        },
                        'x': {'display': False}   # Hide labels for a cleaner 'streaming' look
                    },
                    'plugins': {'legend': {'position': 'top'}}
                }, 
                style={'height': '300px'} # Rigid height to stop stretching
            )
        ])
    ], className="mb-4 shadow border-0")

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="COBBLESTONE ENERGY | REAL-TIME GRID MONITOR", 
        color="dark", dark=True, className="mb-4"
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col(create_chart_card("Gas Plant", "gas-c"), md=6),
            dbc.Col(create_chart_card("Wind Farm", "wind-c"), md=6),
        ]),
        dbc.Row([
            dbc.Col(create_chart_card("Solar Farm", "solar-c"), md=6),
            dbc.Col(create_chart_card("Hydroelectric", "hydro-c"), md=6),
        ]),
        dcc.Interval(id='ui-update', interval=1000) # Refresh every 1 second
    ], fluid=True)
], style={'backgroundColor': '#f4f7f6', 'minHeight': '100vh'})

# --- 4. Dashboard Callbacks ---
@app.callback(
    [Output('gas-c', 'data'), Output('wind-c', 'data'),
     Output('solar-c', 'data'), Output('hydro-c', 'data')],
    [Input('ui-update', 'n_intervals')]
)
def refresh_dashboard(n):
    chart_configs = []
    with data_lock:
        for pt in plant_types:
            entries = list(data_store[pt])
            labels = [str(i) for i in range(len(entries))]
            
            power = [e.get('power_output', 0) for e in entries]
            demand = [e.get('demand', 0) for e in entries]

            chart_configs.append({
                'labels': labels,
                'datasets': [
                    {
                        'label': 'Live Power Output', 
                        'data': power, 
                        'borderColor': '#2ecc71', # Modern Green
                        'backgroundColor': 'rgba(46, 204, 113, 0.1)',
                        'borderWidth': 3,
                        'pointRadius': 0, # Removes dots for a smooth line
                        'fill': True,
                        'tension': 0.3    # Slight curve for aesthetic
                    },
                    {
                        'label': 'Grid Demand', 
                        'data': demand, 
                        'borderColor': '#e74c3c', # Modern Red
                        'borderWidth': 2,
                        'pointRadius': 0,
                        'borderDash': [5, 5]      # Dashed line
                    }
                ]
            })
    return tuple(chart_configs)

if __name__ == '__main__':
    # Start Spark thread
    threading.Thread(target=start_spark_streaming, daemon=True).start()
    # Run Dash
    app.run_server(debug=False, host='0.0.0.0', port=8050)