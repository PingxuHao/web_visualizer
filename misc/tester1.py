import plotly.graph_objects as go
import numpy as np
import plotly.subplots as sp
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

_xs = np.vstack([np.cos(theta), np.sin(theta)])
_ys = A.dot(_xs)

fig = go.Figure()

# Add traces, one for each slider step

for _t in theta:
    _x, _y = getVecs(_t)
    fig.add_trace(
        go.Scatter(
            visible=False, line=dict(color="orange", width=2),
            name="x - unit vector", x=[0, _x[0]], y=[0, _x[1]]))
    fig.add_trace(
        go.Scatter(
            visible=False, line=dict(color="red", width=2),
            name="y - transformed vector", x=[0, _y[0]], y=[0, _y[1]]))
fig.data[0].visible = True
fig.data[1].visible = True


fig.add_trace(go.Scatter(
    visible=True, line=dict(color="orange", width=2), 
    name="Trajectory of x", x=_xs[0], y=_xs[1]))
fig.add_trace(go.Scatter(
    visible=True, line=dict(color="crimson", width=2), 
    name="Trajectory of y", x=_ys[0], y=_ys[1]))
for y in range(-10, 11):
    fig.add_trace(go.Scatter(
        x=[-10, 10],
        y=[y, y],
        mode='lines',
        line=dict(color='lightgray', width=1),
        showlegend=False
    ))
for x in range(-10, 11):
    fig.add_trace(go.Scatter(
        x=[x, x],
        y=[-10, 10],
        mode='lines',
        line=dict(color='lightgray', width=1),
        showlegend=False
    ))

"""
# Add grid lines

# Add transformed grid lines
for i in range(X.shape[0]):
    print(i)
    
    if  i == 10 : #magic number, dont modify
        temp_color = "teal"
        temp_width = 3
    else:
        temp_color = "teal"
        temp_width = 1
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color = temp_color, width=temp_width), 
            x=X_prime[i, :], y=Y_prime[i, :], mode='lines', showlegend=False))
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color = temp_color, width=temp_width), 
            x=X_prime[:, i], y=Y_prime[:, i], mode='lines', showlegend=False))
    
"""
# Add transformed grid lines
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
        args=[{"visible": [False] * Nstp*2 + [True, True, True, True] + [True] * (4*X.shape[0])},  # updated
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

fig.update_layout(
    plot_bgcolor = "black",
    sliders=sliders,
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


# Initialize a list to hold the animation frames
frames = []
# Calculate the number of animation frames
num_frames = 100  # number of frames for the animation

# Calculate the incremental change for each frame
dx = (_ys - _xs) / num_frames
dx_grid = (X_prime - X) / num_frames
dy_grid = (Y_prime - Y) / num_frames
# Initialize a list to hold the animation frames
# Create each frame
for i in range(num_frames):
    # Calculate the current state of the vectors and grid
    _x_current = _xs + dx * i
    X_current = X + dx_grid * i
    Y_current = Y + dy_grid * i

    # Create temporary list to hold the data for this frame
    frame_data = [
        go.Scatter(x=_x_current[0], y=_x_current[1], mode="lines", line=dict(color='orange'))
    ]

    # Add the grid lines to the frame data
    for j in range(X_current.shape[0]):
        frame_data.append(
            go.Scatter(
                x=X_current[j, :],
                y=Y_current[j, :],
                mode='lines', showlegend=False, line=dict(color="teal", width=1)
            )
        )

    for j in range(X_current.shape[1]):
        frame_data.append(
            go.Scatter(
                x=X_current[:, j],
                y=Y_current[:, j],
                mode='lines', showlegend=False, line=dict(color="teal", width=1)
            )
        )

    # Add the current frame to the list
    frames.append(
        go.Frame(
            data=frame_data,
            name=f"frame_{i}"
        )
    )

# Update the figure to include the frames
fig.frames = frames

# Update the layout to include the animation
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 2, "redraw": True}, "fromcurrent": True, "transition": {"duration": 30, "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ],
    sliders=sliders
)

fig.show()