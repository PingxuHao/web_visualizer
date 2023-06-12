import plotly.graph_objects as go
import numpy as np
a11 = 1.0
a12 = -1.0
a21 = -0.25
a22 = 1.0

# Generate a grid in the xy-plane
x = np.linspace(-2.0, 2.0, 10)
y = np.linspace(-2.0, 2.0, 10)
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
            visible=False, line=dict(color="blue", width=2),
            name="x - unit vector", x=[0, _x[0]], y=[0, _x[1]]))
    fig.add_trace(
        go.Scatter(
            visible=False, line=dict(color="red", width=2),
            name="y - transformed vector", x=[0, _y[0]], y=[0, _y[1]]))
fig.data[0].visible = True
fig.data[1].visible = True
fig.add_trace(go.Scatter(
    visible=True, line=dict(color="black", width=1), 
    name="Trajectory of x", x=_xs[0], y=_xs[1]))
fig.add_trace(go.Scatter(
    visible=True, line=dict(color="gray", width=1), 
    name="Trajectory of y", x=_ys[0], y=_ys[1]))

for i in range(X.shape[0]):
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="black", width=1), 
            x=X[i, :], y=Y[i, :]))
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="black", width=1), 
            x=X[:, i], y=Y[:, i]))
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="green", width=1), 
            x=X_prime[i, :], y=Y_prime[i, :]))
    fig.add_trace(
        go.Scatter(
            visible=True, line=dict(color="green", width=1), 
            x=X_prime[:, i], y=Y_prime[:, i]))
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

fig.update_layout(
    sliders=sliders,
    xaxis_title="x",
    yaxis_title="y",
    autosize=False,
    width=800,
    height=800)

fig.update_xaxes(range=[-2.0, 2.0])
fig.update_yaxes(range=[-2.0, 2.0])

fig.show()