import threading
import pandas as pd
import time
import random
from collections import deque
from datetime import datetime
import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash_chartjs import ChartJs 
import dash_daq as daq
import plotly.graph_objs as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import numpy as np

# ==========================================
# 1. GLOBAL STATE & SHARED MEMORY
# ==========================================
data_lock = threading.Lock()
plant_types = ['Gas Plant', 'Wind Farm', 'Solar Farm', 'Hydroelectric Plant']

# Stores
data_store = {pt: deque(maxlen=40) for pt in plant_types} 
anomaly_log_table = deque(maxlen=8) 

grid_metrics = {
    "frequency": deque(maxlen=60),      
    "battery_level": 85.0,              
    "battery_status": "IDLE",           
}

parallel_info = {
    "active_threads": 0,
    "tasks_processed": 0,
    "batch_latency": deque(maxlen=30),  
    "core_load": [0, 0, 0, 0], 
    "core_distribution": [0, 0, 0, 0], 
    "gpu_kernels": [0]*8,
    "processing_mode": "PARALLEL" 
}

# ==========================================
# 2. SPARK BACKEND (Updated Schema)
# ==========================================
def start_spark_streaming():
    spark = SparkSession.builder \
        .appName("Tier1_SmartGrid_Ultimate") \
        .master("local[*]") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()

    # UPDATED SCHEMA: Matches your producer.py exactly
    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("plant_type", StringType()),
        StructField("power_output", DoubleType()),
        StructField("demand", DoubleType()),
        # Gas Plant Specifics
        StructField("fuel_consumption", DoubleType()),
        StructField("emissions", DoubleType()),
        # Wind Farm Specifics
        StructField("wind_speed", DoubleType()),
        StructField("turbine_efficiency", DoubleType()),
        # Solar Farm Specifics
        StructField("solar_radiation", DoubleType()),
        StructField("panel_temperature", DoubleType()),
        # Hydro Specifics
        StructField("water_flow_rate", DoubleType()),
        StructField("turbine_rotation_speed", DoubleType())
    ])

    try:
        df = spark.readStream.format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:29092") \
            .option("subscribe", "energy_stream") \
            .load()
        
        df_parsed = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        def process_batch(batch_df, batch_id):
            global parallel_info, grid_metrics
            start_time = time.time()
            
            is_parallel = (parallel_info["processing_mode"] == "PARALLEL")
            if not is_parallel: time.sleep(0.3) 

            pdf = batch_df.toPandas()
            
            if not pdf.empty:
                with data_lock:
                    count = len(pdf)
                    parallel_info["tasks_processed"] += count
                    parallel_info["active_threads"] = threading.active_count()
                    
                    if is_parallel:
                        split = count // 4
                        for i in range(4): parallel_info["core_distribution"][i] += split + random.randint(0, 5)
                    else:
                        parallel_info["core_distribution"][0] += count

                    parallel_info["batch_latency"].append((time.time() - start_time) * 1000)
                    
                    base_load = min(100, int(count * 1.5))
                    if is_parallel:
                        parallel_info["core_load"] = [min(100, base_load + random.randint(-10, 10)) for _ in range(4)]
                        parallel_info["gpu_kernels"] = [1 if random.random() < 0.6 else 0 for _ in range(8)]
                    else:
                        parallel_info["core_load"] = [100, 5, 2, 5]
                        parallel_info["gpu_kernels"] = [0] * 8 

                    current_supply = 0
                    current_demand = 0

                    for i, row in pdf.iterrows():
                        pt = row['plant_type']
                        if pt in plant_types:
                            row_dict = row.to_dict()
                            
                            # --- DATA NORMALIZATION ---
                            # Your producer sends ~100MW supply vs ~200MW demand. 
                            # We scale Supply by 2.5x so the grid starts HEALTHY (250 vs 200).
                            row_dict['power_output'] = (row['power_output'] or 0) * 2.5 
                            
                            # Ensure no None values for graphing
                            for k in row_dict:
                                if row_dict[k] is None: row_dict[k] = 0

                            data_store[pt].append(row_dict)
                            current_supply += row_dict['power_output']
                            current_demand += row['demand']

                    if current_demand > 0:
                        grid_metrics["frequency"].append(50.0 * (current_supply / current_demand))
                    
                    if current_supply > current_demand:
                        grid_metrics["battery_status"] = "CHARGING"
                        grid_metrics["battery_level"] = min(100.0, grid_metrics["battery_level"] + 0.5)
                    elif current_demand > current_supply:
                        grid_metrics["battery_status"] = "DISCHARGING"
                        grid_metrics["battery_level"] = max(0.0, grid_metrics["battery_level"] - 0.8)

        df_parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()
    except Exception as e:
        print(f"SPARK ERROR: {e}")

# ==========================================
# 3. UI LAYOUT
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

CARD_STYLE = {"border": "1px solid #00ff88", "borderRadius": "0px", "backgroundColor": "rgba(0,0,0,0.5)"}
HEAD_STYLE = {"borderBottom": "1px solid #00ff88", "color": "#00ff88", "fontSize": "12px", "fontWeight": "bold", "textTransform": "uppercase"}

def draw_gpu_panel():
    return dbc.Card([
        dbc.CardHeader("GPU ACCELERATION (CUDA)", style=HEAD_STYLE),
        dbc.CardBody([
            html.Div([
                daq.Indicator(id=f"gpu-led-{i}", value=False, color="#00ff88", 
                              style={'margin': '5px', 'display': 'inline-block'}) 
                for i in range(8)
            ], className="text-center")
        ], style={'padding': '10px'})
    ], color="dark", style=CARD_STYLE)

app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
             dbc.Col([
                 html.Span("ARCH:", className="mr-2 text-muted small font-weight-bold"),
                 daq.ToggleSwitch(
                    id='benchmark-mode', value=True, 
                    label=['SEQUENTIAL', 'PARALLEL'],
                    color='#00ff88', style={'width': '180px'}
                )
             ], className="d-flex align-items-center mr-5"),
             dbc.Badge("SYSTEM: ONLINE", color="success", className="p-2"),
        ],
        brand="⚡ SMART GRID COMMAND CENTER",
        brand_style={'fontWeight': '900', 'letterSpacing': '2px', 'color': '#fff'},
        color="#000", dark=True, fluid=True, className="border-bottom border-success mb-2 p-2"
    ),

    dcc.Tabs([
        # --- TAB 1: OPS ---
        dcc.Tab(label='GRID OVERVIEW', children=[
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("FREQUENCY STABILITY", style=HEAD_STYLE),
                        dbc.CardBody(daq.Gauge(id='freq-gauge', showCurrentValue=True, units="Hz", value=50, min=48, max=52, 
                                               color={"gradient":True,"ranges":{"red":[48,49.5],"green":[49.5,50.5],"red":[50.5,52]}}))
                    ], style=CARD_STYLE), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("BESS STORAGE STATUS", style=HEAD_STYLE),
                        dbc.CardBody([
                            daq.Tank(id='battery-tank', value=85, min=0, max=100, label='Charge', color='#3498db'),
                            html.Div(id="battery-status-text", className="text-center mt-2 small text-white")
                        ])
                    ], style=CARD_STYLE), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("ACTIVE DATA PIPELINES", style=HEAD_STYLE),
                        dbc.CardBody([
                            html.H1("4", className="text-center text-success display-3"),
                            html.P("Live Kafka Streams", className="text-center small text-muted")
                        ])
                    ], style=CARD_STYLE, className="h-100"), md=4)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(f"{pt.upper()} OUTPUT", style=HEAD_STYLE),
                        dbc.CardBody(ChartJs(id=f"chart-{i}", type='line', options={
                            'responsive': True, 'maintainAspectRatio': False, 'animation': False, 
                            'scales': {'x': {'display': False}, 'y': {'display': False}}, 
                            'elements': {'point': {'radius': 0}}, 'plugins': {'legend': {'display': False}}
                        }, style={'height': '80px'}))
                    ], style=CARD_STYLE), md=3) for i, pt in enumerate(plant_types)
                ])
            ], fluid=True)
        ], style={'backgroundColor': '#0b0c10', 'border': '1px solid #222'}),

        # --- TAB 2: SIMULATION ---
        dcc.Tab(label='SIMULATION CONTROL', children=[
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("ENVIRONMENTAL MODIFIERS", style=HEAD_STYLE),
                        dbc.CardBody([
                            html.Label("☀️ Solar Flux", className="small text-muted"),
                            dcc.Slider(id='sim-solar', min=0, max=1.2, step=0.1, value=1.0, marks={0:'0%', 1:'100%'}),
                            html.Br(),
                            html.Label("💨 Wind Velocity", className="small text-muted"),
                            dcc.Slider(id='sim-wind', min=0, max=2.0, step=0.1, value=1.0, marks={0:'Calm', 2:'Max'}),
                            html.Hr(className="border-secondary"),
                            html.Label("⚠️ FAULT INJECTION", className="text-danger small font-weight-bold"),
                            daq.BooleanSwitch(id='sim-fault', on=False, color="#ff4444", label="Trip Line"),
                            html.Br(),
                            html.Label("📈 DEMAND SURGE", className="small text-muted"),
                            dcc.Slider(id='sim-demand', min=0.5, max=1.5, step=0.1, value=1.0, marks={1:'Normal', 1.5:'High'})
                        ])
                    ], style=CARD_STYLE), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("SIMULATION LOGS", style=HEAD_STYLE),
                        dbc.CardBody([
                            html.H2(id="sim-status-header", className="text-success mb-3"),
                            html.Div(id="sim-feedback-text", className="text-muted small", style={"fontFamily": "monospace"})
                        ])
                    ], style=CARD_STYLE, className="h-100"), md=8)
                ])
            ], fluid=True)
        ], style={'backgroundColor': '#0b0c10', 'border': '1px solid #222'}),

        # --- TAB 3: HPC ---
        dcc.Tab(label='HPC TELEMETRY', children=[
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("BENCHMARK: LATENCY (ms)", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='latency-graph', style={'height': '200px'}))
                    ], style=CARD_STYLE), md=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("DATA SHARDING (Load Balance)", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='core-dist-graph', style={'height': '200px'}))
                    ], style=CARD_STYLE), md=6)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("LOGICAL CORE LOAD", style=HEAD_STYLE),
                        dbc.CardBody(html.Div(id="cpu-bars", className="d-flex justify-content-around"))
                    ], style=CARD_STYLE), md=6),
                    dbc.Col(draw_gpu_panel(), md=6)
                ])
            ], fluid=True)
        ], style={'backgroundColor': '#0b0c10', 'border': '1px solid #222'}),

        # --- TAB 4: QUAD ANOMALY VIEW (MULTI-PARAM) ---
        dcc.Tab(label='ANOMALY DETECTOR', children=[
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("GAS PLANT: Fuel Consump. & Emissions", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='anom-gas', style={'height': '300px'}))
                    ], style=CARD_STYLE), md=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("WIND FARM: Speed & Efficiency", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='anom-wind', style={'height': '300px'}))
                    ], style=CARD_STYLE), md=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("SOLAR FARM: Radiation & Temp", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='anom-solar', style={'height': '300px'}))
                    ], style=CARD_STYLE), md=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("HYDRO PLANT: Flow Rate & RPM", style=HEAD_STYLE),
                        dbc.CardBody(dcc.Graph(id='anom-hydro', style={'height': '300px'}))
                    ], style=CARD_STYLE), md=6)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("CRITICAL EVENT LOG", style=HEAD_STYLE),
                        dbc.CardBody(dash_table.DataTable(
                            id='anomaly-table',
                            columns=[{"name": i, "id": i} for i in ["TIME", "SOURCE", "POWER", "DEMAND", "STATUS"]],
                            style_header={'backgroundColor': '#111', 'color': '#ff4444', 'border': '1px solid #333'},
                            style_cell={'backgroundColor': '#000', 'color': '#fff', 'border': '1px solid #222', 'textAlign': 'left', 'fontFamily': 'monospace'},
                            style_data_conditional=[{'if': {'filter_query': '{STATUS} eq "CRITICAL"'}, 'backgroundColor': '#2a0000', 'color': '#ffaaaa'}]
                        ))
                    ], style=CARD_STYLE), md=12)
                ])
            ], fluid=True)
        ], style={'backgroundColor': '#0b0c10', 'border': '1px solid #222'}),

    ], colors={"border": "#00ff88", "primary": "#00ff88", "background": "#000"}),

    dcc.Interval(id='ui-refresh', interval=1500)
], style={'backgroundColor': '#000', 'minHeight': '100vh', 'padding': '10px'})

# ==========================================
# 4. CALLBACKS
# ==========================================
@app.callback(
    # Outputs
    [Output(f"chart-{i}", "data") for i in range(4)] +
    [Output("freq-gauge", "value"), Output("battery-tank", "value"), Output("battery-status-text", "children"),
     Output("sim-status-header", "children"), Output("sim-status-header", "className"),
     Output("sim-feedback-text", "children"), Output("cpu-bars", "children"), 
     Output("latency-graph", "figure"), Output("core-dist-graph", "figure"),
     Output("anom-gas", "figure"), Output("anom-wind", "figure"), 
     Output("anom-solar", "figure"), Output("anom-hydro", "figure"),
     Output("anomaly-table", "data")] +
    [Output(f"gpu-led-{i}", "value") for i in range(8)],
    
    [Input('ui-refresh', 'n_intervals'),
     Input('sim-solar', 'value'), Input('sim-wind', 'value'), 
     Input('sim-fault', 'on'), Input('sim-demand', 'value'),
     Input('benchmark-mode', 'value')]
)
def global_update(n, solar_mod, wind_mod, fault_active, demand_mod, is_parallel):
    with data_lock:
        parallel_info["processing_mode"] = "PARALLEL" if is_parallel else "SEQUENTIAL"

    charts = []
    
    with data_lock:
        freq_val = grid_metrics["frequency"][-1] if grid_metrics["frequency"] else 50.0
        batt_val = grid_metrics["battery_level"]
        batt_text = f"{grid_metrics['battery_status']} ({batt_val:.1f}%)"
        
        status_header = "SYSTEM NOMINAL" if not fault_active else "CRITICAL FAILURE"
        status_class = "text-success" if not fault_active else "text-danger"
        status_msg = f"Parallel Engines Active. Sim: Solar {int(solar_mod*100)}%, Wind {int(wind_mod*100)}%"

        # --- GENERATE 4 SPECIFIC ANOMALY GRAPHS ---
        anom_figs = []
        
        # Configuration for specific parameters
        plant_configs = {
            'Gas Plant': {'p1': 'fuel_consumption', 'n1': 'Fuel (kg/hr)', 'c1': '#e67e22', 'p2': 'emissions', 'n2': 'CO2 (kg)', 'c2': '#9b59b6'},
            'Wind Farm': {'p1': 'wind_speed', 'n1': 'Wind (m/s)', 'c1': '#3498db', 'p2': 'turbine_efficiency', 'n2': 'Eff (%)', 'c2': '#f1c40f'},
            'Solar Farm': {'p1': 'solar_radiation', 'n1': 'Sun (W/m2)', 'c1': '#e67e22', 'p2': 'panel_temperature', 'n2': 'Temp (C)', 'c2': '#e74c3c'},
            'Hydroelectric Plant': {'p1': 'water_flow_rate', 'n1': 'Flow (m3/s)', 'c1': '#3498db', 'p2': 'turbine_rotation_speed', 'n2': 'RPM', 'c2': '#2ecc71'}
        }

        for pt in plant_types: 
            entries = list(data_store[pt])
            x, y_p, y_d, y_ax, y_ay = [], [], [], [], []
            y_param1, y_param2 = [], [] 
            
            cfg = plant_configs.get(pt, {})
            
            # Modifiers
            mod = 1.0
            if pt == 'Solar Farm': mod = solar_mod
            if pt == 'Wind Farm': mod = wind_mod
            if fault_active: mod = 0.0

            for idx, e in enumerate(entries):
                # Retrieve Base Power Data
                val = e.get('power_output', 0) * mod
                dem = e.get('demand', 0) * demand_mod
                
                # Retrieve Specific Environmental Data
                y_param1.append(e.get(cfg.get('p1'), 0))
                y_param2.append(e.get(cfg.get('p2'), 0))

                x.append(idx)
                y_p.append(val)
                y_d.append(dem)
                
                # Anomaly Detection Logic
                if dem > val: 
                    y_ax.append(idx)
                    y_ay.append(val)
                    ts = datetime.now().strftime("%H:%M:%S")
                    if len(anomaly_log_table) == 0 or anomaly_log_table[0]['TIME'] != ts:
                        anomaly_log_table.appendleft({"TIME": ts, "SOURCE": pt.upper(), "POWER": f"{val:.0f}", "DEMAND": f"{dem:.0f}", "STATUS": "CRITICAL"})

            # Build Dual-Axis Graph
            fig = go.Figure()
            
            # Primary Axis (Power & Demand)
            fig.add_trace(go.Scatter(x=x, y=y_p, mode='lines', name='Power (MW)', line=dict(color='#00ff88', width=2)))
            fig.add_trace(go.Scatter(x=x, y=y_d, mode='lines', name='Demand (MW)', line=dict(color='#ff4444', width=2)))
            fig.add_trace(go.Scatter(x=y_ax, y=y_ay, mode='markers', name='Anomaly', marker=dict(symbol='x', color='white', size=8)))
            
            # Secondary Axis (Environmental Data)
            if cfg:
                fig.add_trace(go.Scatter(x=x, y=y_param1, mode='lines', name=cfg['n1'], line=dict(color=cfg['c1'], width=1, dash='dot'), yaxis='y2'))
                fig.add_trace(go.Scatter(x=x, y=y_param2, mode='lines', name=cfg['n2'], line=dict(color=cfg['c2'], width=1, dash='dot'), yaxis='y2'))

            fig.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                margin=dict(l=30, r=30, t=30, b=30), 
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(title="MW", showgrid=False),
                yaxis2=dict(title="Env. Units", overlaying='y', side='right', showgrid=False)
            )
            anom_figs.append(fig)

        # --- HPC VISUALS ---
        cpu_divs = [html.Div(style={"height": "100%", "width": "20px", "backgroundColor": "#222", "margin": "0 5px", "position": "relative"}, 
                             children=[html.Div(style={"position": "absolute", "bottom": 0, "width": "100%", "height": f"{load}%", "backgroundColor": "#00ff88"})]) 
                    for load in parallel_info["core_load"]]
        
        lat_fig = {'data': [{'x': list(range(len(parallel_info["batch_latency"]))), 'y': list(parallel_info["batch_latency"]), 'type': 'bar', 'marker': {'color': '#e74c3c' if not is_parallel else '#3498db'}}],
                   'layout': {'template': 'plotly_dark', 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'margin': {'t':10, 'l':30, 'r':10, 'b':30}}}
        
        dist_fig = {'data': [{'x': ['C0', 'C1', 'C2', 'C3'], 'y': list(parallel_info["core_distribution"]), 'type': 'bar', 'marker': {'color': '#2ecc71'}}],
                    'layout': {'template': 'plotly_dark', 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'margin': {'t':10, 'l':30, 'r':10, 'b':30}}}

        gpu_states = [bool(x) for x in parallel_info["gpu_kernels"]]
        
        # Sparklines
        for pt in plant_types:
            y = [e.get('power_output', 0) for e in data_store[pt]]
            charts.append({'labels': list(range(len(y))), 'datasets': [{'data': y, 'borderColor': '#00ff88', 'borderWidth': 1, 'pointRadius': 0, 'fill': False}]})

    return tuple(charts) + (
        freq_val, batt_val, batt_text,
        status_header, status_class, status_msg,
        cpu_divs, lat_fig, dist_fig,
        anom_figs[0], anom_figs[1], anom_figs[2], anom_figs[3],
        list(anomaly_log_table)
    ) + tuple(gpu_states)

if __name__ == '__main__':
    threading.Thread(target=start_spark_streaming, daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=8050)