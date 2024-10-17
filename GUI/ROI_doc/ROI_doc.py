import os, sys, time
# import atexit
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
from scipy import sparse as ssparse
import threading
import tkinter as tk
from tkinter import filedialog

from caiman.utils.visualization import get_contours

# # Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_of_parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_of_parent_dir)

from placefield_dynamics.neuron_matching import load_data, save_data, set_paths_default

import plotly.express as px
import plotly.graph_objects as go


use_tkinter = False

status = {
    'currentSession': 1,
    'currentNeuronID': None, 
    'clickTime': time.time(),
    'changed': False,
    'loaded': False,
}

status_footprints = {
    'currentCurveNumber': None,
    'currentFootprintID': None,
    'currentSessionID': None,
    'clickTime': time.time(),
    'changed': False,
}

storage = {
    'busy': False,
    'dims': (512, 512),
    'sessions': {}
}

settings = {
    'footprint_display': {
        'intensity_cutoff': 40,     # percentile
        'margin': 30.,              # pixel distance
        'neighbour_distance': 20.,  # pixel distance
        'opacity_base': 0.6,        # opacity of non-selected footprints
        'colormap_active': 'Viridis',
        'colormap_neighbours': 'Inferno',
    },
    'data': {
        'pxtomu': 530.68/512,
    }
}

session_blueprint = {
    'background': None,
    'footprints': None,
    'CoM': None,
    'path': None,
    'S': None,
    'placefield': None
}


fig = go.Figure()
fig.update_layout(
    dragmode=False,  # Disable drag-based interactions like zoom and pan
    coloraxis_showscale=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(visible=False),  # Hide x-axis
    yaxis=dict(visible=False),  # Hide y-axis
    margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
)

fig1 = go.Figure()
fig2 = go.Figure()

# Initialize the Dash app
# app = dash.Dash(__name__)
app = dash.Dash(__name__,prevent_initial_callbacks="initial_duplicate")

### ---------------------- specifying the layout ------------------------
app.layout = html.Div(style={'display':'flex','flex-direction':'row'},children=[
    html.Div(
        style={'width':'80vw'},
        children=[
            html.Div([
                ## header div
                html.H1("VACID"),
                html.P('Visualizing Analysis of Calcium Imaging Data'),
            ]),
            html.Div(
                children=[
                    ## main div, containing plots and data
                    html.Div(
                        style={'display':'flex','flex-direction':'row','justify-content':'space-around'},
                        children=[
                            dcc.Graph(
                                id='main-figure',
                                figure=fig,
                                style={'width': '40vw',
                                       'height': '40vh'},
                                config={
                                    'scrollZoom': False,  # Disable scroll-based zoom
                                    'displayModeBar': True,  # Show the mode bar (can be set to False or 'hover')
                                    'displaylogo': False,  # Remove the Plotly logo from the mode bar
                                    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],  # Remove zoom-related buttons
                                }
                            ),
                    ]),
                    html.Div(
                        style={'display':'flex','flex-direction':'row','justify-content':'space-between','height':'40vh'},
                        children=[
                            dcc.Graph(
                                id='details-graph-1',
                                figure=fig1,
                                style={'width': '30vw'}
                                    #    'height': '30vw',
                                    #    'overflow':'scroll'}
                            ),
                            html.Div(
                                children=[
                                    html.P('Margin:'),
                                    dcc.Input(
                                        id='input_margin',
                                        type='number', 
                                        value=settings['footprint_display']['margin'], 
                                        placeholder='Enter margin',
                                        debounce=True
                                    )
                                ]
                            ),
                            dcc.Graph(
                                id='details-graph-2',
                                figure=fig2,
                                style={'width': '30vw'}
                            )
                    ])
            ]),
    ]),
    html.Div(style={'width':'20vw','background':'green'},children=[
        ## div for session paths, loading data, etc (meta statistics)
        html.Button('Set file path', id='button_setPath', n_clicks=0, disabled=not use_tkinter),
        html.Button('Load data', id='button_loadData', n_clicks=0),
        html.Button('Plot data', id='button_plotData', n_clicks=0),
        html.Div(id='status-div'),  # Output will be shown here
        html.Div(id='click-output', style={'whiteSpace': 'pre-line'}),
    ]),
    # dcc.Store(id='session-data',storage_type='local',data=storage),
    dcc.Store(id='app-status',storage_type='local',data=status),
    dcc.Store(id='status-footprints',storage_type='local',data=status_footprints)
])


### --------------- File dialogues -------------------
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Close the root window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    storage['paths']['sessions'] = file_path  # Store the file path
    storage['busy'] = False  # Reset the flag to indicate dialog is closed


@app.callback(
    Output('status-div', 'children'),
    Input('button_setPath', 'n_clicks')
)
def callback_setPath(n_clicks):
    if n_clicks > 0:
        # Step 7: Trigger tkinter file dialog on the main thread using a shared flag
        storage['busy'] = True
        tkinter_event.set()  # Set the flag to trigger the file dialog
        
        # Wait until the tkinter dialog completes (or is dismissed)
        while storage['busy']:
            time.sleep(0.1)  # Small delay to prevent high CPU usage while waiting
        
        return f'Selected file: {storage["paths"]["sessions"]}'
    return f'No file selected yet.'



### --------------- loading and plotting -------------------
@app.callback(
    Output('app-status', 'data', allow_duplicate=True),
    Input('button_loadData', 'n_clicks'),
    # State('session-data', 'data'),
    State('app-status', 'data'),
)
def callback_button_loadData(n_clicks,status):
    if n_clicks > 0:

        # if not status['loaded']:
        print('loading data')
        mouse_path = '/home/wollex/Documents/Science/WolfGroup/PlaceFields/Programs/data/579ad/'
        pathMatching = os.path.join(mouse_path,'matching/neuron_registration.pkl')
        _, pathsResults = set_paths_default(mouse_path,exclude='redetected')

        # print(pathsResults)

        ld_matching = load_data(pathMatching)

        storage['assignments'] = ld_matching['assignments']
        storage['cm'] = ld_matching['cm']
        storage['shift'] = ld_matching['remap']['shift']

        # print(f'loading session data')
        for s,path in enumerate(pathsResults):
            print(s,path)
            storage['sessions'][s] = session_blueprint.copy()
            storage['sessions'][s]['path'] = path

            ld = load_data(path)
            storage['sessions'][s]['footprints'] = ssparse.lil_matrix((ld['A'].shape[0],storage['assignments'].shape[0]))
            
            idxes_cluster = np.isfinite(storage['assignments'][:,s])
            idxes = storage['assignments'][idxes_cluster,s].astype(int)

            # print(storage['sessions'][s]['footprints'].shape)
            # print(idxes_cluster,idxes)
            storage['sessions'][s]['footprints'][:,idxes_cluster] = ld['A'][:,idxes]#.tolil()
            storage['sessions'][s]['footprints'] = storage['sessions'][s]['footprints'].tocsc()
            storage['sessions'][s]['background'] = ld['Cn']

        status['currentSession'] = 5
        status['loaded'] = True
        
        save_data(storage,'./storage.pkl')
        print('loaded')
        # return 'plotted'
        return status
    # return 'nothing'
    return status


@app.callback(
    Output('main-figure', 'figure'),
    Input('button_plotData', 'n_clicks'),
    # State('session-data', 'data'),
    State('app-status', 'data'),
)
def callback_button_plotData(n_clicks,status):
    if n_clicks > 0:
        storage = load_data('./storage.pkl')

        type_max_val = 2**8-1

        image_data_float = storage['sessions'][status['currentSession']]['background']
        min_val,max_val = np.percentile(image_data_float,[10,95])
        image_data_normalized = np.uint8(np.clip((image_data_float - min_val) / (max_val - min_val) * type_max_val,0,type_max_val))  # Normalize to [0, 255] and convert to uint8
        image_data_normalized = np.tile(image_data_normalized[:,:,np.newaxis],3)
        image_trace = go.Image(z=image_data_normalized)#, zmin=[0,0,0], zmax=[255,255,255], hoverinfo='none')  # Create an image trace
        fig.add_trace(image_trace)

        fig.update_layout(
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
        )

        # image_data_normalized = np.uint8((image_data_float - image_data_float.min()) / (image_data_float.max() - image_data_float.min()) * 255)  # Normalize to [0, 255] and convert to uint8
        # print(updated_fig)
        # image_trace = go.Image(z=image_data_normalized)#, zmin=[0,0,0], zmax=[255,255,255], hoverinfo='none')  # Create an image trace
        # updated_fig.add_trace(
        #     image_trace
        #     # px.imshow(storage['sessions'][status['currentSession']]['background'],origin='lower')
        # )

        contours = get_contours(storage['sessions'][status['currentSession']]['footprints'][:,:10],storage['dims'])
        for neuronID,c in enumerate(contours):
            print(neuronID,status['currentSession'],storage['assignments'][neuronID,status['currentSession']])
            
            if np.isnan(storage['assignments'][neuronID,status['currentSession']]):
                continue
            
            fig.add_trace(go.Scatter(
                x=c['coordinates'][:,0],
                y=c['coordinates'][:,1],  # For demonstration, this should be the contour points
                mode='lines',
                line=dict(width=2,color='red'),
                hovertemplate=f"Neuron ID: {neuronID}<br>CoM: ({c['CoM'][0]:.2f},{c['CoM'][1]:.2f})<br><extra></extra>",  # Custom 
                # hovertemplate=f"Neuron {n}<br>X: {c['CoM'][0]:.1f}<br>Y: {c['CoM'][1]:.1f}<br><extra></extra>",  # Custom hover info
                showlegend=False,
                customdata=[neuronID]  # Custom data for context menu
            )
        )

        return fig
    return dash.no_update



@app.callback(
    [Output('app-status', 'data', allow_duplicate=True),
     Output('click-output', 'children')],
    Input('main-figure', 'clickData'),
    State('app-status', 'data'),
    prevent_initial_call=True
)
def callback_click_contours_neuronContour(clickData,data):
    if clickData is None:
        return dash.no_update, 'Click on a neuron to see the details.'

    # Extract information about the clicked point
    point_data = clickData['points'][0]
    curve_number = point_data['curveNumber']
    
    if curve_number==0:
        return dash.no_update, 'Click on a neuron to see the details.' 
    
    # Extract custom data from the clicked point
    data['currentNeuronID'] = fig['data'][curve_number]['customdata'][0]
    data['clickTime'] = time.time()
    data['changed'] = True

    custom_info = f"You clicked on neuron {data['currentNeuronID']}"  # Get the customdata attached to the point
    return data, custom_info


### ------------------- interaction with neurons --------------------
def get_neighbouring_neurons(c,neighbor_distance=20):
    
    storage = load_data('./storage.pkl')

    cm_px = storage['cm'] * settings['data']['pxtomu']
    ## obtain average center of mass of neuron
    cm = np.nanmean(cm_px[c,...],axis=0)

    ## obtain neuronID and sessionID of neighbouring neurons
    idxes_c,idxes_s = np.where(np.sqrt((cm_px[...,0] - cm[0])**2 + (cm_px[...,1] - cm[1])**2) < neighbor_distance)

    idxes_same = np.where(idxes_c==c)[0]
    idxes_c = np.delete(idxes_c,idxes_same)
    return np.unique(idxes_c)
    # idxes_s = np.delete(idxes_s,idxes_same)

    # return idxes_c, idxes_s


def plot_footprint_3d(neuronID,sessionID,storage,colormap,opacity):

    footprintID = int(storage['assignments'][neuronID,sessionID])
    A = storage['sessions'][sessionID]['footprints'][:,neuronID]
    thr = np.percentile(A.data[A.data>0],settings['footprint_display']['intensity_cutoff'])
    thr = A.data[A.data>0].mean()

    X,Y = np.unravel_index(A.indices,storage['dims'])
    display_idxes = A.data>thr
    A = A/A.max()*0.6

    return go.Mesh3d(
        z=sessionID+A.data[display_idxes],
        x=(X[display_idxes]+storage['shift'][sessionID,1])*settings['data']['pxtomu'],
        y=(Y[display_idxes]+storage['shift'][sessionID,0])*settings['data']['pxtomu'],
        intensity=A.data[display_idxes],  # Color-coding based on intensity
        colorscale=colormap,        # You can choose any predefined colorscale
        opacity=opacity,
        # hoverinfo='all',
        # hoverlabel=dict(margin=dict(t=5, b=5, l=5, r=5)),
        hovertemplate=f"Session {sessionID}<br>Neuron ID: {neuronID}<br>Footprint ID: {footprintID}<br><extra></extra>",  # Custom hover info
        showscale=False,
        customdata=[sessionID,footprintID],
    )


@app.callback(
    Output('app-status', 'data', allow_duplicate=True),
    Input('input_margin', 'value'),
    State('app-status', 'data'),
)
def update_margin(value,data):
    settings['footprint_display']['margin'] = value
    data['changed'] = True
    return data


# @app.callback(
#     Output('app-status', 'data', allow_duplicate=True),
#     Input('details-graph-1', 'dblClickData'),
#     State('app-status', 'data'),
# )
# def callback_dblclick_footprints(dblClickData,data):
#     # print('dblclick!!')
#     # print(dblClickData)
#     # data['currentNeuronID'] = None
#     data['changed'] = True
#     return data

    
@app.callback(
    [Output('details-graph-1','figure', allow_duplicate=True),
     Output('app-status', 'data')],
    Input('app-status', 'data'),
    State('details-graph-1', 'figure'),
    prevent_initial_call=True
)
def plot_footprints(data,fig1):

    if data['currentNeuronID'] is None:
        return fig1

    storage = load_data('./storage.pkl')

    neuronID = int(data['currentNeuronID'])
    cm_neuron = np.nanmean(storage['cm'][neuronID,...],axis=0)

    # print('selected neuron ID:',neuronID)

    
    updated_fig = go.Figure(fig1)

    ## plot all footprints of the selected neuron
    for sessionID in np.where(np.isfinite(storage['assignments'][neuronID,:]))[0]:
        updated_fig.add_trace(plot_footprint_3d(neuronID,sessionID,storage,colormap=settings['footprint_display']['colormap_active'],opacity=settings['footprint_display']['opacity_base']))
    
    ## plot all neighbouring footprints
    neighbourIDs= get_neighbouring_neurons(neuronID,settings['footprint_display']['neighbour_distance'])
    for neuronID in neighbourIDs:
        for sessionID in np.where(np.isfinite(storage['assignments'][neuronID,:]))[0]:
            updated_fig.add_trace(plot_footprint_3d(neuronID,sessionID,storage,colormap=settings['footprint_display']['colormap_neighbours'],opacity=settings['footprint_display']['opacity_base']*1/2.))  


    margin = settings['footprint_display']['margin']
    # print(cm_neuron)
    # print('x:',cm_neuron[1] + np.array([-margin,+margin]))
    # print('y:',cm_neuron[0] + np.array([-margin,+margin]))
    updated_fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=True,
                range=cm_neuron[1] + np.array([-margin,+margin]),
            ),  # Disable x-axis grid and ticks
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=True,
                range=cm_neuron[0] + np.array([-margin,+margin]),
            ),  # Disable y-axis grid and ticks
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Disable z-axis grid and ticks
            dragmode='turntable',  # Optional: Set a rotation mode for better 3D interaction
            aspectratio=dict(x=1, y=1, z=2)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        # height=1200,
        hovermode="y",
        # title="2D Plot with Custom X-Axis Limits"
    )

    data['changed'] = False
    # Display the clicked coordinates and custom info
    return updated_fig, data



@app.callback(
    Output('status-footprints', 'data'),
    Input('details-graph-1', 'clickData'),
    [State('status-footprints', 'data'),
    State('details-graph-1', 'figure')],    # for some reason needs to be passed explicitly
    prevent_initial_call=True
)
def callback_click_footprint_3d(clickData,data,fig1):

    # print(data)
    # if clickData is None:
    #     return dash.no_update 

    print(time.time(),data['clickTime'],time.time()-data['clickTime'])
    # if data['clickTime'] is not None and (time.time() - data['clickTime'] < 0.5):
    #     print('this appears to be a double click')

    # data['clickTime'] = time.time()


    # Extract information about the clicked point
    point_data = clickData['points'][0]
    curve_number = point_data['curveNumber']
    
    if curve_number==0:
        return dash.no_update
    
    # Extract custom data from the clicked point
    data['currentCurveNumber'] = curve_number
    data['currentSessionID'], data['currentFootprintID'] = fig1['data'][curve_number]['customdata']
    data['clickTime'] = time.time()
    data['changed'] = True

    return data


# @app.callback(
#     Output('details-graph-1', 'figure'),
#     #  Output('click-output', 'children')],
#     Input('details-graph-1', 'clickData'),
#     [State('details-graph-1', 'figure'),
#     State('details-graph-1', 'relayoutData')],
#     prevent_initial_call=True  # Do not run until a click happens
# )
@app.callback(
    Output('details-graph-1', 'figure'),
    Input('status-footprints', 'data'),
    State('details-graph-1', 'figure'),
    prevent_initial_call=True
)
def callback_click_matched_neuronFootprint(clickData,fig1,relayoutdata):
    if clickData is None:
        return dash.no_update, "Click on a trace to see its custom data."
    
    # Extract the curveNumber (index of the clicked trace) and the customdata
    # custom_info = clickData['points'][0]['customdata']  # Custom data attached to the trace
    
    # Update the opacity of the clicked trace (lower opacity for others)
    curve_number = clickData['points'][0]['curveNumber']
    updated_traces = []
    for i, trace in enumerate(fig1['data']):
        if i == curve_number:
            # Change the opacity of the clicked trace to highlight it
            trace.update(opacity=1.0)  # Full opacity for the clicked trace
        else:
            trace.update(opacity=0.6)  # Reduce opacity for others
        updated_traces.append(trace)
    
    # Update the figure with modified traces
    fig1['data'] = updated_traces
    updated_fig = go.Figure(fig1)
    if 'scene.camera' in relayoutdata:
        updated_fig.update_layout(scene_camera=relayoutdata['scene.camera'])
    
    # Return the updated figure and display the custom data
    # return updated_fig, f'You clicked on bla'#{custom_info}'
    return updated_fig#, f'You clicked on bla'#{custom_info}'




# def on_startup(mouse_path):
#     if not os.path.exists("startup.lock"):
#         with open("startup.lock", "w") as f:
#             f.write("Startup code executed")
#     else:
#         return
#     # print(mouse_path)
    

# def on_exit():
    # if os.path.exists("startup.lock"):
    #     os.remove("startup.lock")
    # print('exiting')
# atexit.register(on_exit)


### --------------------- opening the app ---------------------
if use_tkinter:
    # Step 8: Event to synchronize tkinter actions on the main thread
    tkinter_event = threading.Event()

    # Step 9: Main thread will listen for tkinter_event and open the file dialog
    def main_thread_tkinter():
        while True:
            tkinter_event.wait()  # Wait until the flag is set by the Dash thread
            open_file_dialog()  # Open tkinter file dialog
            tkinter_event.clear()  # Reset the event after dialog is closed

    # Step 10: Start the main thread to handle tkinter dialogs
    tkinter_thread = threading.Thread(target=main_thread_tkinter)
    tkinter_thread.start()


    def run_dash():
        app.run_server(debug=True, use_reloader=False)  # Running Dash app in a separate thread

    # Step 11: Start Dash in a background thread
    dash_thread = threading.Thread(target=run_dash)
    dash_thread.start()

    # Step 12: Keep the main thread alive for tkinter interactions
    tkinter_thread.join()
else:
    # Run the app
    if __name__ == '__main__':
        # print('running dash in main thread...')
        # on_startup(mouse_path)
        app.run_server(debug=True)#,use_reloader=False)#, use_reloader=True

        # if os.path.exists("startup.lock"):
        #     os.remove("startup.lock")