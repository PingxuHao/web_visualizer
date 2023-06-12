import plotly.graph_objects as go
import numpy as np
a11 = 1.0
a12 = -1.0
a21 = -0.25
a22 = 1.0
A = np.array([
    [a11, a12],
    [a21, a22]])

theta = np.linspace(0,2*np.pi,361)[:-1]
Nstp = len(theta)

def getVecs(t):
    _x = [np.cos(t), np.sin(t)]
    _y = A.dot(_x)
    return _x, _y

_xs = np.vstack([np.cos(theta), np.sin(theta)])
_ys = A.dot(_xs)

fig2 = go.Figure()

# Add traces, one for each slider step
for _t in theta:
    _x, _y = getVecs(_t)
    fig2.add_trace(
        go.Scatter(
            visible=False, line=dict(color="blue", width=2),
            name="x - unit vector", x=[0, _x[0]], y=[0, _x[1]]))
    fig2.add_trace(
        go.Scatter(
            visible=False, line=dict(color="red", width=2),
            name="y - transformed vector", x=[0, _y[0]], y=[0, _y[1]]))
fig2.data[0].visible = True
fig2.data[1].visible = True
fig2.add_trace(go.Scatter(
    visible=True, line=dict(color="black", width=1), 
    name="Trajectory of x", x=_xs[0], y=_xs[1]))
fig2.add_trace(go.Scatter(
    visible=True, line=dict(color="gray", width=1), 
    name="Trajectory of y", x=_ys[0], y=_ys[1]))

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

fig2.update_layout(
    sliders=sliders,
    xaxis_title="x",
    yaxis_title="y",
    autosize=False,
    width=800,
    height=800)

fig2.update_xaxes(range=[-2.0, 2.0])
fig2.update_yaxes(range=[-2.0, 2.0])
