import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from testin_text import testing_str1
from fig1 import fig1

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Visualizer v1.2'

def subpage_1_layout():
    
    return html.Div([
        html.H1('Title1',
                ),
        dcc.Graph(figure=fig1,
                  style={'width': '100%', 
                        'display':'block', 
                        'margin-left': 'auto',
                        'margin-right': 'auto'}),
        html.P(testing_str1,
               style={'width': '100%', 
                        'display':'block', 
                        'margin-left': 30,
                        'margin-right': 30}),
        html.Img(src=app.get_asset_url('image/Solar_System.jpg'), 
                style={'width': '80%', 
                    'display':'block', 
                    'margin-left': 'auto',
                    'margin-right': 'auto'})
    ])


def subpage_2_layout():
    from fig2 import fig2
    return html.Div([
        html.H1('Titl32'),
        dcc.Graph(figure=fig2)
    ])
    
def subpage_3_layout():
    return html.Div([
        html.Video(src=app.get_asset_url('cat.mp4'), controls=True,
                    style={'width': '80%', 
                    'display':'block', 
                    'margin-left': 'auto',
                    'margin-right': 'auto'}),
        dcc.Graph(figure=fig1,
                  style={'width': '100%', 
                        'display':'block', 
                        'margin-left': 'auto',
                        'margin-right': 'auto'}),
        html.A("for more information on linear transformation click here", 
               href="https://en.wikipedia.org/wiki/Linear_map",
               style={'width': '100%', 
                        'display':'block', 
                        'margin-left': 'auto',
                        'margin-right': 'auto'})
    ])

app.layout = html.Div([
    dbc.Navbar([
        dbc.DropdownMenu(
            label="Menu",
            id="menu-dropdown",
            children=[
                dbc.DropdownMenuItem("Subpage 1", id="subpage-1-link"),
                dbc.DropdownMenuItem("Subpage 2", id="subpage-2-link"),
                dbc.DropdownMenuItem("Subpage 3", id="subpage-3-link")
            ],
        )
    ], sticky='top'),
    html.Div(id='content', children=[
        # Preload subpage content and hide by default
        html.Div(id='subpage-1', children=subpage_1_layout(), style={'display': 'none'}),
        html.Div(id='subpage-2', children=subpage_2_layout(), style={'display': 'none'}),
        html.Div(id='subpage-3', children=subpage_3_layout(), style={'display': 'none'})
    ])
])

@app.callback(
    Output('subpage-1', 'style'),
    Output('subpage-2', 'style'),
    Output('subpage-3', 'style'),
    [Input('subpage-1-link', 'n_clicks'),
     Input('subpage-2-link', 'n_clicks'),
     Input('subpage-3-link', 'n_clicks')]
)
def show_subpage(subpage1_clicks, subpage2_clicks, subpage3_clicks):
    ctx = dash.callback_context
    
    style_to_show = {'display': 'block'}
    style_to_hide = {'display': 'none'}
    
    if not ctx.triggered:
        return style_to_hide, style_to_hide, style_to_hide
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'subpage-1-link':
        return style_to_show, style_to_hide, style_to_hide
    elif triggered_id == 'subpage-2-link':
        return style_to_hide, style_to_show, style_to_hide
    # Check if subpage 3 was clicked
    elif triggered_id == 'subpage-3-link':
        return style_to_hide, style_to_hide, style_to_show
    
    return style_to_hide, style_to_hide, style_to_hide


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')