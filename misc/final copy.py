import plotly.graph_objects as go
import numpy as np
a11 = 1.0
a12 = -1.0
a21 = -0.25
a22 = 1.0

# Generate grid
x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
X, Y = np.meshgrid(x, y)


A = np.array([
    [a11, a12],
    [a21, a22]])

# Flatten the X and Y grids to apply the transformation matrix
XY = np.vstack([X.flatten(), Y.flatten()])

# Apply the transformation matrix
XY_prime = A.dot(XY)

# Unflatten the transformed grid
X_prime, Y_prime = XY_prime[0].reshape(X.shape), XY_prime[1].reshape(Y.shape)

theta = np.linspace(0,2*np.pi,361)[:-1]
Nstp = len(theta)

def getVecs(t):
    _x = [np.cos(t), np.sin(t)]
    _y = A.dot(_x)
    return _x, _y
#circle
"""_xs = np.vstack([np.cos(theta), np.sin(theta)])
"""

#square
_xs = [[0, 0, 1, 1, 0],[0, 1, 1, 0, 0]]
_ys = A.dot(_xs)

fig = go.Figure()

for _t in theta:
    _x, _y = getVecs(_t)

fig.add_trace(go.Scatter(
    visible=True, line=dict(color="red", width=1), 
    name="Trajectory of x", x=_xs[0], y=_xs[1]))
fig.add_trace(go.Scatter(
    visible=True, line=dict(color="orange", width=1), 
    name="Trajectory of y", x=_ys[0], y=_ys[1]))

for i in range(X.shape[0]):
    X_line = X[i, :]
    Y_line = Y[i, :]
    XY_line = np.vstack([X_line.flatten(), Y_line.flatten()])
    XY_line_prime = A.dot(XY_line)
    X_line_prime, Y_line_prime = XY_line_prime[0].reshape(X_line.shape), XY_line_prime[1].reshape(Y_line.shape)
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="teal", width=1), 
            x=X_line_prime, y=Y_line_prime, mode='lines', showlegend=False
        )
    )
    
for i in range(X.shape[1]):
    X_line = X[:, i]
    Y_line = Y[:, i]
    XY_line = np.vstack([X_line.flatten(), Y_line.flatten()])
    XY_line_prime = A.dot(XY_line)
    X_line_prime, Y_line_prime = XY_line_prime[0].reshape(X_line.shape), XY_line_prime[1].reshape(Y_line.shape)
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="teal", width=1), 
            x=X_line_prime, y=Y_line_prime, mode='lines', showlegend=False
        )
    )
# Create and add slider
steps = []
for i in range(Nstp):
    _x, _y = getVecs(theta[i])
    _yl = np.linalg.norm(_y)
    _ya = np.arctan2(_y[1], _y[0])/(2*np.pi)*360
    _xa = np.arctan2(_x[1], _x[0])/(2*np.pi)*360
    step = dict(
        method="update",
        args=[{"visible": [False] * Nstp*2 + [True, True]},
              {"title": f"|y|={_yl:4.3f}, dir(x)-dir(y)="+f"{_xa-_ya:4.1f} deg"}],
        label=f"{theta[i]/(2*np.pi)*360:3.2f} deg"
    )
    for j in [0,1]:
        step["args"][0]["visible"][2*i+j] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "theta = "},
    pad={"t": 50},
    steps=steps
)]

#grap design
fig.update_layout(
    plot_bgcolor = "black",
    xaxis_title="x",
    yaxis_title="y",
    autosize=False,
    width=800,
    height=800)
fig.update_xaxes(
    showgrid = True,
    #showline=True,
    gridwidth = 1,
    dtick=0.5,
    zeroline=True,  # Display the zero line on the y-axis
    zerolinecolor='lightgrey',
    zerolinewidth = 1,
    gridcolor='grey',range=[-2.0, 2.0])
fig.update_yaxes(
    showline=True,
    #showgrid = True,
    gridwidth = 1,
    dtick=0.5,
    zeroline=True,  # Display the zero line on the y-axis
    zerolinecolor='lightgrey',
    zerolinewidth = 1,
    gridcolor='grey',range=[-2.0, 2.0])


frames = []
num_frames = 50 
dx = (_ys - _xs) / num_frames #change in circle/square
dx_grid = (X_prime - X) / num_frames  #change in x axis
dy_grid = (Y_prime - Y) / num_frames  #change in y axis

for i in range(num_frames):
    # current state of the vectors and grid
    _x_current = _xs + dx * i
    X_current = X + dx_grid * i
    Y_current = Y + dy_grid * i
    
    # list to hold frame
    frame_data = [
        go.Scatter(x=_x_current[0], 
                   y=_x_current[1], 
                   mode="lines", 
                   line=dict(color='orange',width = 3),
                   marker=dict(color='orange',size=6)
                   )
    ]
    frame_data .extend ([
        go.Scatter(
            x=[_x_current[0][j]], 
            y=[_x_current[1][j]], 
            mode="markers", 
            marker=dict(color='orange',size=6),
            name=f'Corner {j+1}'
            ) for j in range(4)
    ])
    
    #add what is already on the graph, add in the list below
    frame_data.extend([
        go.Scatter(
            visible=True, line=dict(color="crimson", width=2),
            name="Trajectory of x", x=_xs[0], y=_xs[1])
    ])
    # Add the grid lines to the frame data
    for j in range(X_current.shape[0]):
        X_line = X_current[j,:]
        Y_line = Y_current[j,:] 
        frame_data.append(
            go.Scatter(
                x=X_line,
                y=Y_line,
                mode='lines', line=dict(color="teal")
            )
        )
    for j in range(X_current.shape[0]):
        X_line = X_current[:,j]
        Y_line = Y_current[:,j] 
        frame_data.append(
            go.Scatter(
                x=X_line,
                y=Y_line,
                mode='lines', line=dict(color="teal")
            )
        )
    frames.append(
        go.Frame(
            data=frame_data,
            name=f"frame_{i}"
        )
    )
fig.frames = frames
# Update button and slider
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
            ],
            "type": "buttons"
        }
    ],
    sliders = sliders
)

fig.show()