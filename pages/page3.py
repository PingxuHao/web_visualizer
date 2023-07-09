from dash import Dash, dcc, html, Input, Output, register_page, dcc, callback, State #, callback # If you need callbacks, import it here.

register_page(
    __name__,
    name='Page 3',
    top_nav=True,
    path='/page3'
)


layout = html.Div([
        html.H1(id = 'tank', children = "Page 3"),
        html.Button('Linear Transformation', id='abc', n_clicks=0)
    ])




@callback(
    Output ('tank', 'children'),
    Input ('abc', 'value')
)

def call_but (e):
    return "cbb"