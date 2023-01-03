# imports
from dash import Dash, html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from numpy import pi
import plotly.graph_objs as go
import math

mapbox_access_token = 'pk.eyJ1IjoiaC1yLWdoYW5iYXJpIiwiYSI6ImNsMmZ3YnRzbDBldHozYm56MXBnaWdyY3YifQ.v8KZrsYNGTfg8x67b-sOVA'
#--------------------------------------------------------------------------------------
#Preparing the dataframes
print("reading sector db file. please wait...")
sector_df = pd.read_excel("Sector_DB.xlsx")

print("reading KPI file. please wait...")
kpi_df = pd.read_excel("2GKPI.xlsx", converters={'Date': pd.to_datetime})


KPI_List = [
    k for k in kpi_df.columns if k not in ("BSC", "CELL", "REGION", "PROVINCE", "LAC", "CID", "Date")
]

print("reading ticket file. please wait...")
ticket_df = pd.read_csv("Clarity_09.18.2022.csv")

ticket_df['REPORTED'] = pd.to_datetime(ticket_df['REPORTED'])

# Keeping only the required data
NE_List = ['BTS', 'ENODEB', 'NODEB', 'WBTS']
ticket_df = ticket_df[ticket_df['EQUP_EQUT_ABBREVIATION'].isin(NE_List)]
ticket_df["Issue_Type"] = "KPI_TT"

Avail_KPI = ['TCH_Availability(Nokia_SEG)', 'Cell_Availability_including_blocked_by_user_state(Nokia_UCell)',
             'Cell_Availability_Rate_Include_Blocking(UCell_Eric)', 'TCH_Availability(HU_Cell)',
             'TCH_Availability(Eric_Cell)', 'Radio_Network_Availability_Ratio(Hu_Cell)',
             'cell_availability_include_manual_blocking(Nokia_LTE_CELL_NO_NULL)',
             'Cell_Availability_Rate_Include_Blocking(Cell_EricLTE_NO_NULL)',
             'Cell_Availability_Rate_include_Blocking(Cell_Hu_NO_NULL)'
             ]

ticket_df.loc[ticket_df['DEGRADED_KPI'].isin(Avail_KPI), 'Issue_Type'] = "Avail_TT"
ticket_df = ticket_df.reset_index(drop=True)

# Extracting Site_ID from EQUP_INDEX column, adding Index column and merging with Lat and Long columns
temp = np.where((ticket_df["EQUP_INDEX"].apply(len) > 10),
                ticket_df['EQUP_INDEX'].str.split('|'), #Use the output of this if True
                ticket_df['EQUP_INDEX'].str.split('_') #Else use this.
                )

dft = pd.DataFrame(temp.tolist(), columns=["A", "Site"]) #create a new dataframe with the columns we need
dft['Site'] = dft['Site'].str.strip()

dft.fillna("XX", inplace=True)

#Extracting Site ID
site_temp = np.where((dft["Site"].apply(len) < 8),
                dft['Site'].str[:6], #Use the output of this if True
                dft['Site'].str[:2] + dft['Site'].str[4:8] #Else use this.
                )

site_temp = pd.DataFrame(site_temp.tolist()) # create a new dataframe with the columns we need
ticket_df["Site"] = site_temp
ticket_df["Cell"] = dft['Site']


#Adding Province Index columns
ticket_df["Index"] = ticket_df["Site"].str[:2]
sector_df["Index"] = sector_df["Site"].str[:2]

site_df = sector_df[['Site', 'LATITUDE', 'LONGITUDE', 'Index']].drop_duplicates(subset=['Site'])

ticket_df = ticket_df.merge(site_df[['Site', 'LATITUDE', 'LONGITUDE']], how='left', on='Site')

# dropping rows with no latitude and longitude
ticket_df.dropna(subset=['LATITUDE'], inplace=True)

#Extracting Sector ID
sector_temp = np.where((dft["Site"].apply(len) < 8),
                dft['Site'].str[:], #Use the output of this if True
                dft['Site'].str[:2] + dft['Site'].str[4:9] #Else use this.
                )

sector_temp = pd.DataFrame(sector_temp.tolist()) # create a new dataframe with the columns we need
ticket_df["Sector"] = sector_temp

# print(ticket_df.shape)

ticket_df = ticket_df.merge(sector_df[['Sector', 'AZIMUTH', 'TA', 'Coverage']], how='left', on='Sector')
ticket_df['TA'] = ticket_df['TA'].round(3)


#ticket_df.to_csv('test_output.csv', index=False)

avail_df = ticket_df[ticket_df["Issue_Type"]=="Avail_TT"]

ticket_df = ticket_df[ticket_df["Issue_Type"]=="KPI_TT"]

# filling missing values
# sectors without azimuth will be -1
ticket_df['AZIMUTH'].fillna(-1, inplace=True)

#--------------------------------------------------------------------------------------
# BOOTSTRAP+, CERULEAN+, COSMO+, SKETCHY, SLATE+, UNITED
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#--------------------------------------------------------------------------------------
# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 64,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "padding": "2rem 1rem",
    "background-color": "#EAEAEA",
    "overflow": "scroll",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
}

# padding for the page content
CONTENT_STYLE = {
    "position": "fixed",
    "top": 64,
    "right": 0,
    "bottom": 0,
    "width": "69.4rem",
    "padding": "0.5rem 0.5rem",
    "background-color": "#EAEAEA", #f8f9fa
    "overflow": "scroll",
}

NAVBAR_STYLE = {
    "top": 0,
    "left": 0,
    "right": 0,
    "height": 50,
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("SHAYGAN", className="display-5"),
        html.Hr(),
        html.P(
            "Data Analytics & Machine Learning", className="lead"
        ),
    ],
    style=SIDEBAR_STYLE,
)

# search_bar = dbc.Row(
#     [
#         dbc.Col(dbc.Input(type="search", placeholder="Search")),
#         dbc.Col(
#             dbc.Button(
#                 "Search", color="primary", className="ms-2", n_clicks=0
#             ),
#             width="auto",
#         ),
#     ],
#     className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
#     align="center",
# )

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("Data Driven Services", className="lead ms-2 fs-3 text"), width=10),
                        #dbc.Col(dbc.NavbarBrand("shaygan-tele", className="lead ms-2 fs-5 text"), width=2),
                    ],
                    align="center",
                    className="g-0",
                ),
                style={"textDecoration": "none", "font-style": "italic"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                # search_bar,
                # id="navbar-collapse",
                # is_open=False,
                # navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    fixed="top",
    style=NAVBAR_STYLE,
)

card = dbc.Card(
    [
        dbc.CardHeader("Ticket Count", className="lead",  style={'font-size': '20px',
                                                                 'background-color': 'white',
                                                                 'text-align': 'center',
                                                                 }),
        dbc.CardBody(
            [
                html.H2("34", style={'font-size': '58px',
                                     'text-align': 'center',
                                     'color': 'white',
                                     }),
            ]
        ),
    ],
    style={'background-color': 'orange',},
)

# functions for drawing sector
def degree2rad(degrees):
    return degrees * pi / 180


def sec_poly(long, lat, bearing, radius=0.5, vbw=60):
    R = 6378.1  # Radius of the Earth
    rad_bearing = degree2rad(bearing)

    site_lat = math.radians(lat)  # site lat point converted to radians
    site_lon = math.radians(long)  # site long point converted to radians

    coords = []
    n = 5
    t = np.linspace(degree2rad(bearing - (vbw / 2)), degree2rad(bearing + (vbw / 2)), n)
    for brg in t:
        bor_lat = math.asin(math.sin(site_lat) * math.cos(radius / R) + math.cos(site_lat) * math.sin(radius / R) * math.cos(brg))
        bor_lon = site_lon + math.atan2(math.sin(brg) * math.sin(radius / R) * math.cos(site_lat),
                                    math.cos(radius / R) - math.sin(site_lat) * math.sin(bor_lat))

        bor_lat = math.degrees(bor_lat)
        bor_lon = math.degrees(bor_lon)

        coords.append([bor_lon, bor_lat])

    coords.insert(0, [long, lat])
    coords.append([long, lat])

    return (coords)
#--------------------------------------------------------------------------------------
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(navbar, width=12)
    ]),

    html.Div([


        dbc.Row([
            # Line chart for ticket trend
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label('Trend of Radio Tickets', className="lead")
                    ], width=3),

                    dbc.Col([
                        dcc.Dropdown([x for x in sorted(ticket_df["Index"].unique())],
                                     multi=False,
                                     placeholder='Province',
                                     id='pro_line_dpdn',
                                     style={
                                         'borderWidth': '0px',
                                         'font-size': '13px'},
                                     ),
                    ], width=2, ),

                    dbc.Col([
                        dcc.Dropdown([x for x in sorted(ticket_df["DEGRADED_KPI"].unique())],
                                     multi=False,
                                     id='kpi_line_dpdn',
                                     placeholder='KPI',
                                     style={
                                         'borderWidth': '0px',
                                         'font-size': '13px'},
                                     ),
                    ], width=7, ),

                ], className='mt-1'),

                dbc.Row([
                    html.Hr(),
                    #dcc.Loading(children=[dcc.Graph(id='line_tt_trend', config={'displayModeBar': False})], color="#119DFF", type="cube", fullscreen=True,),
                    dcc.Graph(id='line_tt_trend', config={'displayModeBar': False})
                ], className='ms-1 me-1'),
            ], width=9, className='shadow-sm bg-white rounded ms-3 me-2'),
            # Card right side of line chart
            dbc.Col([
                dbc.Row([card], className='shadow-sm rounded'),
                html.Br(),
                #dbc.Row([card], className='shadow-sm rounded'),
            ],width=2, className='ms-3'),

        ],),

        html.Br(),
        # Bar chart for ticket per province distribution
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    html.Label('Tickets per Province', className="lead", ),
                ], width=3),

                dbc.Col([
                    dcc.DatePickerSingle(
                        id='bar_date_picker',  # ID to be used for callback
                        first_day_of_week=6,  # Display of calendar when open (0 = Sunday)
                        clearable=True,  # whether or not the user can clear the dropdown
                        placeholder='Select a Date',
                        number_of_months_shown=1,  # number of months shown when calendar is open
                        min_date_allowed=ticket_df["REPORTED"].min(),
                        # minimum date allowed on the DatePickerRange component
                        max_date_allowed=ticket_df["REPORTED"].max(),
                        # maximum date allowed on the DatePickerRange component
                        initial_visible_month=ticket_df["REPORTED"].max(),
                        # the month initially presented when the user opens the calendar
                        display_format='MMM Do, YY',
                        # how selected dates are displayed in the DatePickerRange component.
                        month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.

                    ),

                ], width=2,),

                dbc.Col([
                    dcc.Dropdown([x for x in sorted(ticket_df["DEGRADED_KPI"].unique())],
                                 multi=False,
                                 id='kpi_bar_dpdn',
                                 placeholder='KPI',
                                 style={
                                     'borderWidth': '0px',
                                     'font-size': '13px'},
                                 ),
                ], width=6, ),

            ], className='mt-1'),

            dbc.Row([
                html.Hr(),
                dcc.Graph(id='bar_province', config={'displayModeBar': False})
            ], className='ms-1'),
        ], className='shadow-sm bg-white rounded ms-1 me-1'),

        html.Br(),
        # Map View
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    html.Label('Map View', className="lead", ),
                ], width=3),

                dbc.Col([
                    dcc.DatePickerSingle(
                        id='map_date_picker',  # ID to be used for callback
                        first_day_of_week=6,  # Display of calendar when open (0 = Sunday)
                        clearable=True,  # whether or not the user can clear the dropdown
                        placeholder='Select a Date',
                        number_of_months_shown=1,  # number of months shown when calendar is open
                        min_date_allowed=ticket_df["REPORTED"].min(),
                        # minimum date allowed on the DatePickerRange component
                        max_date_allowed=ticket_df["REPORTED"].max(),
                        # maximum date allowed on the DatePickerRange component
                        initial_visible_month=ticket_df["REPORTED"].max(),
                        # the month initially presented when the user opens the calendar
                        display_format='MMM Do, YY',
                        # how selected dates are displayed in the DatePickerRange component.
                        month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.

                    ),

                ], width=2, ),

                dbc.Col([
                    dcc.Dropdown([x for x in sorted(ticket_df["Index"].unique())],
                                 multi=False,
                                 id='pro_map_dpdn',
                                 placeholder='Province',
                                 value='KH',
                                 style={
                                     'borderWidth': '0px',
                                     'font-size': '13px'},
                                 ),
                ], width=2, ),

                dbc.Col([
                    dcc.Dropdown(sorted(['open-street-map', 'carto-positron', 'carto-darkmatter', 'stamen-terrain', 'stamen-toner',
                                        'stamen-watercolor', 'basic', 'streets', 'outdoors', 'light', 'dark', 'satellite', 'satellite-streets']),
                                 multi=False,
                                 id='type_map_dpdn',
                                 placeholder='Map Style',
                                 value='light',
                                 style={
                                     'borderWidth': '0px',
                                     'font-size': '13px'},
                                 ),
                ], width=2, ),

            ], className='mt-1'),

            dbc.Row([
                html.Hr(),
                dcc.Graph(id='map_sites', config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False}),
            ], className='ms-0'),
        ], className='shadow-sm bg-white rounded ms-1 me-1'),
        html.Br(),
        # KPI monitoring line chart
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    html.Label('KPI Monitoring', className="lead mb-1", ),
                ], width=2),

                dbc.Col([
                    html.Div(
                        [
                            dbc.Button("Filter", id="kpi-moni-offc-btn", color="secondary", size="sm", n_clicks=0),
                            dbc.Offcanvas(

                                [
                                    dbc.Col(
                                        [
                                            dbc.RadioItems(
                                                id="kpi-interval-radios",
                                                className="btn-group",
                                                inputClassName="btn-check",
                                                labelClassName="btn btn-outline-secondary",
                                                labelCheckedClassName="active",
                                                options=[
                                                    {"label": "1D", "value": 1},
                                                    {"label": "2D", "value": 2},
                                                    {"label": "3D", "value": 3},
                                                    {"label": "1W", "value": 7},
                                                    {"label": "2W", "value": 14},
                                                    {"label": "3W", "value": 21},
                                                ],
                                                value=1,
                                            ),
                                        ],
                                        width={"offset": 2}, className="radio-group",
                                    ),
                                    dcc.Dropdown([x for x in sorted(kpi_df["CELL"].unique())],
                                                 multi=False,
                                                 placeholder='Select Cell',
                                                 id='cell_moni_dpdn',
                                                 style={
                                                     'borderWidth': '0px',
                                                     'font-size': '13px'},
                                                 ),
                                    dcc.Dropdown(sorted(KPI_List),
                                                 multi=False,
                                                 id='kpi_moni_dpdn',
                                                 placeholder='Select KPI',
                                                 style={
                                                     'borderWidth': '0px',
                                                     'font-size': '13px'},
                                                 ),
                                ],
                                id="kpi-moni-offc",
                                title="KPI Monitoring Filters",
                                placement='end',
                                is_open=False,
                            ),
                        ]
                    ),
                         ], width={'size': 1}), #"offset": 9}

            ], className='mt-1'),

            dbc.Row([
                html.Hr(),
                dcc.Graph(id='line_kpi_trend', config={'displayModeBar': True, 'displaylogo': False})
            ], className='ms-0'),
        ], className='shadow-sm bg-white rounded ms-1 me-1'),
        # end of graphs
        html.Br(),
    ], style=CONTENT_STYLE),

    dbc.Row([
        dbc.Col(sidebar, width=2)
    ]),

], fluid=True, )
#--------------------------------------------------------------------------------------
#                                         callbacks
#--------------------------------------------------------------------------------------
# KPI monitoring offcanvas
@app.callback(
    Output("kpi-moni-offc", "is_open"),
    Input("kpi-moni-offc-btn", "n_clicks"),
    [State("kpi-moni-offc", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open
#--------------------------------------------------------------------------------------
# Line Chart for KPI trend
@app.callback(
    Output(component_id='line_kpi_trend', component_property='figure'),
    [Input(component_id='kpi-interval-radios', component_property='value'),
     Input(component_id='cell_moni_dpdn', component_property='value'),
     Input(component_id='kpi_moni_dpdn', component_property='value')])

def update_graph(chosen_interval, chosen_cell, chosen_kpi):

    if chosen_cell is not None and chosen_kpi is not None:
        kpi_moni_df = kpi_df[kpi_df["CELL"] == chosen_cell]
        kpi_moni_df = kpi_moni_df[['Date', chosen_kpi]]
        kpi_moni_df = kpi_moni_df.sort_values(by=['Date'])
        kpi_moni_df = kpi_moni_df.tail(24*chosen_interval)


    else:
        kpi_moni_df = kpi_df


    # if chosen_kpi is not None:
    #     kpi_moni_df = kpi_moni_df[['Date', chosen_kpi]]
    # preparing tickets' trend line chart
    kpi_moni_fig=px.line(kpi_moni_df, x='Date', y=chosen_kpi, markers=False)# height=450

    kpi_moni_fig.update_traces(hovertemplate='%{y}', line=dict(color='rgb(77, 207, 241)', width=1),)

    kpi_moni_fig.update_layout(
        xaxis=dict(
            title='',
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title=chosen_kpi,
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        margin={'l':0,'t':10,'b':0,'r':10},
        plot_bgcolor='white',
        hovermode="x",
        )

    return (kpi_moni_fig)
#--------------------------------------------------------------------------------------
# Line Chart for tickets' trend
@app.callback(
    Output(component_id='line_tt_trend', component_property='figure'),
    [Input(component_id='pro_line_dpdn', component_property='value'),
     Input(component_id='kpi_line_dpdn', component_property='value')])

def update_graph(chosen_province, chosen_kpi):

    line_df = ticket_df
    if chosen_province is not None:
        line_df = line_df[line_df["Index"] == chosen_province]
    if chosen_kpi is not None:
        line_df = line_df[line_df["DEGRADED_KPI"] == chosen_kpi]


    # preparing data for tickets' trend line chart
    line_df = line_df.groupby(['REPORTED'], as_index=False)['PROM_NUMBER'].count()
    line_df = line_df.set_index('REPORTED')
    line_df = line_df.groupby([pd.Grouper(freq="D")])['PROM_NUMBER'].sum().reset_index()

    # preparing tickets' trend line chart
    line_fig=px.line(line_df, x="REPORTED", y="PROM_NUMBER", markers=False)# height=450

    line_fig.update_traces(hovertemplate='%{y}', line_shape='spline', line=dict(color='rgb(77, 207, 241)', width=1),)

    line_fig.update_layout(
        xaxis=dict(
            title='',
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='Number of Tickets',
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        margin={'l':0,'t':10,'b':0,'r':10},
        plot_bgcolor='white',
        hovermode="x",
        )

    return (line_fig)
#--------------------------------------------------------------------------------------
# Bar Chart for tickets per province
@app.callback(
    Output(component_id='bar_province', component_property='figure'),
    [Input(component_id='bar_date_picker', component_property='date'),
     Input(component_id='kpi_bar_dpdn', component_property='value')])

def update_graph(chosen_date, chosen_kpi):
    bar_df = ticket_df
    if chosen_date is not None:
        bar_df = ticket_df[ticket_df['REPORTED'].dt.strftime('%Y-%m-%d') == chosen_date]
    if chosen_kpi is not None:
        bar_df = bar_df[bar_df['DEGRADED_KPI'] == chosen_kpi]

    bar_df = bar_df.groupby(['Index', 'DEGRADED_KPI'], as_index=False)['PROM_NUMBER'].count()

    barchart=px.bar(
            data_frame=bar_df,
            x='Index',
            y='PROM_NUMBER',
            color="DEGRADED_KPI",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            height=450
            )
    barchart.update_traces(hovertemplate='%{y}')
    barchart.update_layout(
        xaxis=dict(
            title='',
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='Number of Tickets',
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        margin={'l':0,'t':10,'b':0,'r':10},
        plot_bgcolor='white',
        showlegend=False,
    )
    return (barchart)
#--------------------------------------------------------------------------------------
# Mapbox for site locations
@callback(Output('map_sites', 'figure'),
          [Input('map_date_picker', 'date'),
           Input('pro_map_dpdn', 'value'),
           Input('type_map_dpdn', 'value'),
           Input("map_sites", "clickData")])

def update_figure(chosen_date ,chosen_province, chosen_map, click_data):
    poly_sec_list = []
    map_df = ticket_df
    avail_map_df = avail_df
    site_map_df = site_df
    if chosen_date is not None:
        map_df = map_df[map_df['REPORTED'].dt.strftime('%Y-%m-%d') == chosen_date]
        avail_map_df = avail_map_df[avail_map_df['REPORTED'].dt.strftime('%Y-%m-%d') == chosen_date]
    if chosen_province is not None:
        map_df = map_df[map_df['Index'] == chosen_province]
        avail_map_df = avail_map_df[avail_map_df['Index'] == chosen_province]
        site_map_df = site_map_df[site_map_df['Index'] == chosen_province]
    if click_data is not None:
        poly_sec_list = sec_poly(long=click_data["points"][0]["customdata"][29], lat=click_data["points"][0]["customdata"][28], bearing=click_data["points"][0]["customdata"][31], radius=click_data["points"][0]["customdata"][32])
        #print(poly_sec_list)
    if chosen_map is not None:
        map_style = chosen_map
    else:
        map_style = 'white-bg'

    fig = go.Figure()

    # showing sites on map
    fig.add_trace(go.Scattermapbox(
        lat=site_map_df['LATITUDE'],
        lon=site_map_df['LONGITUDE'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=8,
            color='rgb(0, 0, 255)',
            opacity=1
        ),
        text="",
        hoverinfo='none',
        name="Sites",
        unselected={'marker': {'opacity': 1, 'size': 8}},
        selected={'marker': {'opacity': 0.5, 'size': 20}},
    ))

    # showing Avail. tickets on map
    fig.add_trace(go.Scattermapbox(
        lat=avail_map_df['LATITUDE'],
        lon=avail_map_df['LONGITUDE'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='rgb(255, 0, 0)',
            opacity=1
        ),
        text="",
        hoverinfo='none',
        name="Availability",
        unselected={'marker': {'opacity': 1, 'size': 15}},
        selected={'marker': {'opacity': 0.5, 'size': 30}},
    ))

    # showing KPI tickets on map
    fig.add_trace(go.Scattermapbox(
        lat=map_df['LATITUDE'],
        lon=map_df['LONGITUDE'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color='rgb(255, 255, 0)',
            opacity=1
        ),
        name="KPI Issue",
        unselected={'marker': {'opacity': 1, 'size': 10}},
        selected={'marker': {'opacity': 0.5, 'size': 15}},
        hoverinfo='text',
        customdata=map_df,
        hovertemplate=
        "NE: %{customdata[26]}<br>" +
        "KPI: %{customdata[5]}<br>" +
        "Root Cause: %{customdata[14]}<br>" +
        "Alram Count: %{customdata[18]}<br>" +
        "TA(Km): %{customdata[32]}<br>" +
        "Coverage Type: %{customdata[33]}<br>" +
        "Azimuth: %{customdata[31]}<extra></extra>",
    ))

    fig.update_layout(
        uirevision='foo',  # preserves state of figure/map after callback activated
        showlegend=True,
        clickmode='event+select',
        hovermode='closest',
        hoverdistance=2,
        height=500,
        margin={'l': 0, 't': 0, 'b': 5, 'r': 0},
        legend=dict(
            x=0,
            y=1,
            title="",
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            font=dict(color='blue')
        ),
        mapbox=dict(
            accesstoken=mapbox_access_token,
            layers=
            [{
                'source': {
                    'type': "FeatureCollection",
                    'features': [{
                        'type': "Feature",
                        'geometry': {
                            'type': "MultiPolygon",
                            'coordinates': [[poly_sec_list]]
                        }
                    }]
                },
                'type': "fill", 'below': "traces", 'color': "green", 'opacity': 0.3}],
            style=map_style,
            bearing=0,
            center=dict(
                lat=map_df['LATITUDE'].mean(),
                lon=map_df['LONGITUDE'].mean(),
            ),
            pitch=0,
            zoom=7
        ),
    )

    return fig
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=False)