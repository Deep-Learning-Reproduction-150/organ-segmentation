from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


base_url = "https://raw.githubusercontent.com/plotly/datasets/master/ply/"
mesh_names = ["sandal", "scissors", "shark", "walkman"]
dataframes = {name: pd.read_csv(base_url + name + "-ply.csv") for name in mesh_names}

organ_names = [f"organ_{i}" for i in range(9)] + ["all_organs"]
for organ in organ_names:
    try:
        dataframes[organ] = pd.read_csv(f"{organ}.csv")
    except FileNotFoundError:
        pass

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("PLY Object Explorer"),
        html.P("Choose an object:"),
        dcc.Dropdown(id="dropdown", options=mesh_names + organ_names, value="sandal", clearable=False),
        dcc.Graph(id="graph"),
    ]
)


@app.callback(Output("graph", "figure"), Input("dropdown", "value"))
def display_mesh(name):
    df = dataframes[name]  # replace with your own data source
    if "i" in df.columns:
        fig = go.Figure(
            go.Mesh3d(
                x=df.x,
                y=df.y,
                z=df.z,
                i=df.i,
                j=df.j,
                k=df.k,
                facecolor=df.facecolor,
            )
        )
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="organ_names")
    return fig


app.run_server(debug=True)
