import plotly.graph_objects as go
import numpy as np
a11 = 1
a12 = -1
a21 = 1
a22 = 2
linear = False
unit_v = np.array([1,1])
A = np.array([
    [a11, a12],
    [a21, a22]])
U,S,V = np.linalg.svd(A)
A1 = U
A2 = np.diag(S)
A3 = V

# Generate grid
x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
X, Y = np.meshgrid(x, y)
# Flatten the X and Y grids to apply the transformation matrix
XY = np.vstack([X.flatten(), Y.flatten()])

# Apply the transformation matrix
XY_prime = A.dot(XY)
unit_v_prime = A.dot(unit_v)
# Unflatten the transformed grid
X_prime, Y_prime = XY_prime[0].reshape(X.shape), XY_prime[1].reshape(Y.shape)

# Apply the rotation matrix
XY_prime_U = A1.dot(XY)
unit_v_prime_U = A1.dot(unit_v)
# Unflatten the transformed grid
X_prime_U, Y_prime_U = XY_prime_U[0].reshape(X.shape), XY_prime_U[1].reshape(Y.shape)

# Apply the stretch matrix
XY_prime_S = A2.dot(XY)
unit_v_prime_S = A2.dot(unit_v)
# Unflatten the transformed grid
X_prime_S, Y_prime_S = XY_prime_S[0].reshape(X.shape), XY_prime_S[1].reshape(Y.shape)

# Apply the rotation matrix 2
XY_prime_V = A3.dot(XY)
unit_v_prime_V = A3.dot(unit_v)
# Unflatten the transformed grid
X_prime_V, Y_prime_V = XY_prime_V[0].reshape(X.shape), XY_prime_V[1].reshape(Y.shape)


#square
_xs = [[0, 0, 1, 1, 0],[0, 1, 1, 0, 0]]
_ys = A.dot(_xs)


# Initial data for the fig1ure
_x_current = _xs
X_current = X
Y_current = Y
initial_data = [
    go.Scatter(x=_x_current[0], 
               y=_x_current[1], 
               mode="lines", 
               line=dict(color='orange',width = 3),
               marker=dict(color='orange',size=6)
               )
]
initial_data.extend([
    go.Scatter(
        x=[_x_current[0][j]], 
        y=[_x_current[1][j]], 
        mode="markers", 
        marker=dict(color='orange',size=6),
        name=f'Corner {j+1}'
        ) for j in range(4)
])

# Add what is already on the graph, add in the list below
initial_data.extend([
    go.Scatter(
        visible=True, line=dict(color="crimson", width=2),
        name="Trajectory of x", x=_xs[0], y=_xs[1])
])
# Add the grid lines to the frame data
for j in range(X_current.shape[0]):
    X_line = X_current[j,:]
    Y_line = Y_current[j,:] 
    initial_data.append(
        go.Scatter(
            x=X_line,
            y=Y_line,
            mode='lines+markers', line=dict(color="teal")
        )
    )
for j in range(X_current.shape[0]):
    X_line = X_current[:,j]
    Y_line = Y_current[:,j] 
    initial_data.append(
        go.Scatter(
            x=X_line,
            y=Y_line,
            mode='lines+markers', line=dict(color="teal")
        )
    )
#arrow body
initial_data.append(
    go.Scatter(
                x=[0,1], 
                y=[0,1], 
                mode="lines",
                line=dict(color='red', width=2),
                showlegend=False
            ))
arrow_angle = np.pi / 6  # arrow angle in radians, adjust as needed
arrow_length = 0.2  # arrow length, adjust as needed
v_current = unit_v
arrow_plus = unit_v - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) + arrow_angle),
                                                      np.sin(np.arctan2(v_current[1], v_current[0]) + arrow_angle)])
arrow_minus = unit_v - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) - arrow_angle),
                                                    np.sin(np.arctan2(v_current[1], v_current[0]) - arrow_angle)])
#arrow head
initial_data.append(
    go.Scatter(
        x=[v_current[0], arrow_plus[0],  arrow_minus[0],v_current[0]], 
        y=[v_current[1], arrow_plus[1],  arrow_minus[1],v_current[1]],
        mode="lines",
        line=dict(color='red', width=2),
        showlegend=False,
        fill='toself',  # Fill the area inside the lines
        fillcolor='red'
            ))
fig1 = go.Figure(data=initial_data)
#grap design
#change fig siez here
fig1.update_layout(
    plot_bgcolor = "black",
    xaxis_title="x",
    yaxis_title="y",
    autosize=False,
    width=800,
    height=800)
fig1.update_xaxes(
    showgrid = True,
    #showline=True,
    gridwidth = 1,
    dtick=1,
    zeroline=True,  # Display the zero line on the y-axis
    zerolinecolor='lightgrey',
    zerolinewidth = 1,
    gridcolor='grey',range=[-3, 3])
fig1.update_yaxes(
    showline=True,
    #showgrid = True,
    gridwidth = 1,
    dtick=1,
    zeroline=True,  # Display the zero line on the y-axis
    zerolinecolor='lightgrey',
    zerolinewidth = 1,
    gridcolor='grey',range=[-3, 3])

#animation
#//////////////////////////////////////////////////////////////////////////////////////////
frames = []
total_num_frames = 0

def add_rotation_frame(A, num_frames, square, X, Y, unit_v, total_frames):
    loc_frames = []
    _xs = square
    _ys = A.dot(_xs)
    v_prime = A.dot(unit_v)
    XY = np.vstack([X.flatten(), Y.flatten()])

    angle_initial = np.arctan2(unit_v[1], unit_v[0])
    
    angle_final = np.arctan2(v_prime[1], v_prime[0])
    dtheta = (angle_final - angle_initial) / (num_frames - 1)

    for i in range(num_frames):
        theta =  dtheta * i
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        XY_prime = R.dot(XY)
        unit_v_prime = R.dot(unit_v)
        X_prime, Y_prime = XY_prime[0].reshape(X.shape), XY_prime[1].reshape(Y.shape)
        _x_current = R.dot(_xs)
        X_current = X_prime
        Y_current = Y_prime
        v_current = unit_v_prime

        arrow_plus = v_current - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) + arrow_angle),
                                                          np.sin(np.arctan2(v_current[1], v_current[0]) + arrow_angle)])
        arrow_minus = v_current - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) - arrow_angle),
                                                            np.sin(np.arctan2(v_current[1], v_current[0]) - arrow_angle)])

        frame_data = [
            go.Scatter(x=_x_current[0], 
                    y=_x_current[1], 
                    mode="lines", 
                    line=dict(color='orange',width = 3),
                    marker=dict(color='orange',size=6)
                    )
        ]
        # Add the grid lines to the frame data
        for j in range(X_current.shape[0]):
            X_line = X_current[j,:]
            Y_line = Y_current[j,:] 
            frame_data.append(
                go.Scatter(
                    x=X_line,
                    y=Y_line,
                    mode='lines+markers', line=dict(color="teal")
                )
            )
        for j in range(X_current.shape[0]):
            X_line = X_current[:,j]
            Y_line = Y_current[:,j] 
            frame_data.append(
                go.Scatter(
                    x=X_line,
                    y=Y_line,
                    mode='lines+markers', line=dict(color="teal")
                )
            )

        #origional orange square vetex 
        frame_data.extend ([
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
        # arrow line
        frame_data.append(go.Scatter(
            x=[0, v_current[0]], 
            y=[0, v_current[1]], 
            mode="lines",
            line=dict(color='red', width=2),
            showlegend=False
        ))
        
        # arrow head
        # fill not working here for arrow head
        frame_data.append(go.Scatter(
            x=[v_current[0], arrow_plus[0],  arrow_minus[0],v_current[0]], 
            y=[v_current[1], arrow_plus[1],  arrow_minus[1],v_current[1]],
            mode="lines",
            line=dict(color='red', width=2),
            showlegend=False,
            fill='toself',  # Fill the area inside the lines
            fillcolor='red'
            )
        )
        
        loc_frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame_{total_frames + i}"
            )
        )
    return [_ys,X_prime, Y_prime,unit_v_prime,loc_frames]
    




def add_frame (A,num_frames,square,X,Y,unit_v,total_frames):
    loc_frames = []
    #square
    _xs = square
    _ys = A.dot(_xs)
    # Flatten the X and Y grids to apply the transformation matrix
    XY = np.vstack([X.flatten(), Y.flatten()])

    # Apply the transformation matrix
    XY_prime = A.dot(XY)
    unit_v_prime = A.dot(unit_v)
    # Unflatten the transformed grid
    X_prime, Y_prime = XY_prime[0].reshape(X.shape), XY_prime[1].reshape(Y.shape)
    
    dx = (_ys - _xs) / (num_frames - 1) #change in circle/square
    dx_grid = (X_prime - X) / (num_frames - 1)  #change in x axis
    dy_grid = (Y_prime - Y) / (num_frames - 1)  #change in y axis
    dv = (unit_v_prime - unit_v) / (num_frames - 1) #change in unit vect

    for i in range(num_frames):
        # current state of the vectors and grid
        _x_current = _xs + dx * i
        X_current = X + dx_grid * i
        Y_current = Y + dy_grid * i
        v_current = unit_v + dv*i
        # arrow head directions
        arrow_plus = v_current - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) + arrow_angle),
                                                        np.sin(np.arctan2(v_current[1], v_current[0]) + arrow_angle)])
        arrow_minus = v_current - arrow_length * np.array([np.cos(np.arctan2(v_current[1], v_current[0]) - arrow_angle),
                                                        np.sin(np.arctan2(v_current[1], v_current[0]) - arrow_angle)])
        # list to hold frame
        # square
        frame_data = [
            go.Scatter(x=_x_current[0], 
                    y=_x_current[1], 
                    mode="lines", 
                    line=dict(color='orange',width = 3),
                    marker=dict(color='orange',size=6)
                    )
        ]
        # Add the grid lines to the frame data
        for j in range(X_current.shape[0]):
            X_line = X_current[j,:]
            Y_line = Y_current[j,:] 
            frame_data.append(
                go.Scatter(
                    x=X_line,
                    y=Y_line,
                    mode='lines+markers', line=dict(color="teal")
                )
            )
        for j in range(X_current.shape[0]):
            X_line = X_current[:,j]
            Y_line = Y_current[:,j] 
            frame_data.append(
                go.Scatter(
                    x=X_line,
                    y=Y_line,
                    mode='lines+markers', line=dict(color="teal")
                )
            )

        #origional orange square vetex 
        frame_data.extend ([
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
        # arrow line
        frame_data.append(go.Scatter(
            x=[0, v_current[0]], 
            y=[0, v_current[1]], 
            mode="lines",
            line=dict(color='red', width=2),
            showlegend=False
        ))
        
        # arrow head
        # fill not working here for arrow head
        frame_data.append(go.Scatter(
            x=[v_current[0], arrow_plus[0],  arrow_minus[0],v_current[0]], 
            y=[v_current[1], arrow_plus[1],  arrow_minus[1],v_current[1]],
            mode="lines",
            line=dict(color='red', width=2),
            showlegend=False,
            fill='toself',  # Fill the area inside the lines
            fillcolor='red'
            )
        )
        
        loc_frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame_{total_frames + i}"
            )
        )
    return [_ys,X_prime, Y_prime,unit_v_prime,loc_frames]
        
x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
X1, Y1 = np.meshgrid(x, y)
square1 =  [[0, 0, 1, 1, 0],[0, 1, 1, 0, 0]]
if not linear:       
    l1 = add_rotation_frame(A1,50,square1,X1,Y1,unit_v,total_num_frames)
    total_num_frames += 50
    frame1 = l1[4]

    l2 = add_frame(A2,50,l1[0],l1[1],l1[2],l1[3],total_num_frames)
    total_num_frames += 50
    frame2 = l2[4]

    l3 = add_rotation_frame(A3,50,l2[0],l2[1],l2[2],l2[3],total_num_frames)
    total_num_frames += 50
    frame3 = l3[4]
else:
    l1 = add_frame(A1,50,square1,X1,Y1,unit_v,total_num_frames)
    total_num_frames += 50
    frame1 = l1[4]

    l2 = add_frame(A2,50,l1[0],l1[1],l1[2],l1[3],total_num_frames)
    total_num_frames += 50
    frame2 = l2[4]

    l3 = add_frame(A3,50,l2[0],l2[1],l2[2],l2[3],total_num_frames)
    total_num_frames += 50
    frame3 = l3[4]
frames.extend (frame1)
frames.extend (frame2)
frames.extend (frame3)
fig1.frames = frames

# Update button and slider
sliders = [{
    'pad': {"b": 10, "t": 60},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': [{
        'args': [[f'frame_{i}'], {'frame': {'duration': 1, 'redraw': False},
                                  'mode': 'immediate',
                                  'transition': {'duration': 10}}],
        'label': str(i + 1), 'method': 'animate'} for i in range(total_num_frames)        
              ],
    'currentvalue': {'font': {'size': 20}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'}
}]

fig1.update_layout(updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 1, "redraw": False}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate"
            },
        ],
        "type": "buttons"
    }],
    sliders=sliders
)
#fig1.show()