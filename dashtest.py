from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from glob import glob


data_files = glob("./data/examples/*.csv")
print(f"Found files {data_files}")
mesh_names = [file.split("/")[-1] for file in data_files]

dataframes = {}
for organ in data_files:
    dataframes[organ] = pd.read_csv(organ)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("PLY Object Explorer"),
        html.P("Choose an object:"),
        dcc.Dropdown(id="dropdown", options=list(dataframes.keys()), value=list(dataframes.keys())[0], clearable=False),
        dcc.Graph(id="graph"),
    ]
)


@app.callback(Output("graph", "figure"), Input("dropdown", "value"))
def display_mesh(name):
    df = dataframes[name]  # replace with your own data source

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="organ_names")
    return fig


app.run_server(debug=True)
