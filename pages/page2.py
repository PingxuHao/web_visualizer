from dash import Dash, dcc, html, Input, Output, register_page, dcc, callback, State #, callback # If you need callbacks, import it here.
from .fig1 import fig1_func,f1t

register_page(
    __name__,
    name='Page 2',
    top_nav=True,
    path='/page2'
)
def layout():
    layout = html.Div([
        html.H1("Page 2"),
        dcc.Input(id="a11", type="number", placeholder="a11", style={'marginRight':'10px'}),
        dcc.Input(id="a12", type="number", placeholder="a12", style={'marginRight':'10px'}),
        dcc.Input(id="a21", type="number", placeholder="a21", style={'marginRight':'10px'}),
        dcc.Input(id="a22", type="number", placeholder="a22", style={'marginRight':'10px'}),
        html.Button('Linear Transformation', id='submit-button', n_clicks=0),
        dcc.Graph(
            id='graph1',
            figure = fig1_func(),
            style={
                'width': '100%', 
                'display':'block', 
                'margin-left': 'auto',
                'margin-right': 'auto'
            }
        )  
    ])
    return layout




@callback(
    Output(component_id='graph1', component_property='figure'),
    [Input (component_id='submit-button', component_property= 'n_clicks')],
    [
        State(component_id='a11', component_property='value'),
        State(component_id='a12', component_property='value'),
        State(component_id='a21', component_property='value'),
        State(component_id='a22', component_property='value')
    ]
)
def update_graph(n, a11_val, a12_val, a21_val, a22_val):
    return fig1_func(a11_val, a12_val, a21_val, a22_val)