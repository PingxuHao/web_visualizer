from dash import Dash, dcc, html, Input, Output, register_page, dcc, callback, State 

from .fig1 import fig1_func, f1t

register_page(
    __name__,
    name='Page 2',
    top_nav=True,
    path='/page2'
)
temp_fig = {}
def layout():
    layout = dcc.Loading(
        id="loading",
        type="default",  # you can choose from 'default', 'circle' or 'cube'
        children=[
            html.Div([
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
        ]
    )
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
    # Only update the figure when the button has been clicked
    if n > 0:
        return fig1_func(a11_val, a12_val, a21_val, a22_val)
    else:
        # Return an empty figure when the page initially loads
        return fig1_func()
