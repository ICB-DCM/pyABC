import base64
import io
import math
import os
import pathlib
import re
import tempfile
from pathlib import Path

import click
import dash
import dash_bootstrap_components as dbc
import matplotlib
import matplotlib.pyplot as plt
from dash import dcc, html
from dash.dependencies import Input, Output

import pyabc
from pyabc.storage import history as h

matplotlib.use('Agg')

static = str(pathlib.Path(__file__).parent.absolute()) + '/assets/'
os.chdir('../../')

static_logo = Path(str(pathlib.Path(__file__).parent.absolute())).parent.parent
DOWNLOAD_DIR = tempfile.mkdtemp() + '/'
db_path = DOWNLOAD_DIR
parameter = ""
square_png = static + 'square_v2.png'
square_base64 = base64.b64encode(open(square_png, 'rb').read()).decode('ascii')
pyABC_png = str(static_logo) + '/doc/logo/logo.png'
pyABC_base64 = base64.b64encode(open(pyABC_png, 'rb').read()).decode('ascii')
para_list = []
colors = {
    'div': '#595959',
}
# Initialise the app
app = dash.Dash(__name__)

# Define the app
app.layout = app.layout = html.Div(
    children=[
        html.Div(
            id='alert_div',
        ),
        html.Div(
            className='row',  # Define the row element
            children=[
                # header
                html.Div(
                    className='four columns div-user-controls',
                    children=[
                        html.Img(
                            id='pyABC_logo',
                            src='data:image/png;base64,{}'.format(
                                pyABC_base64
                            ),
                            width="300",
                        ),
                        html.Br(),
                        html.H3('(Visulization Web server)'),
                        html.Hr(),
                        html.Div(
                            [
                                "A copy of the generated files will be"
                                " save in: ",
                                dcc.Input(
                                    id="download_path",
                                    placeholder=DOWNLOAD_DIR,
                                    type='text',
                                    value=DOWNLOAD_DIR,
                                ),
                            ]
                        ),
                        html.Div(id="hidden-div", style={"display": "none"}),
                        dcc.Input(
                            id='upload-data',
                            type='text',
                            placeholder='Please write the path to the db '
                            'file HERE',
                            # children=html.Div(
                            #     [
                            #         'Drag and Drop or ',
                            #         html.A('Select Files'),
                            #     ]
                            # ),
                            style={
                                'width': '70%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'display': 'inline-block',
                            },
                            multiple=True,
                        ),
                        html.Button(
                            'Upload',
                            id='upload_file',
                            style={'height': '60px'},
                            n_clicks=0,
                        ),
                        # show details in boxes once db is loaded
                        html.Div(
                            [
                                html.Div(id='output-data-upload'),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                "ABC runs: ",
                                                dcc.Dropdown(id='ABC_runs'),
                                            ]
                                        ),
                                    ],
                                    style={'width': '100%', 'height': '100%'},
                                ),
                            ]
                        ),
                    ],
                ),  # Define the left element
                html.Div(
                    id='information_grid',
                    className='eight columns div-for-charts bg-grey',
                    style={'height': '1300px'},
                ),
                # Define the right element
            ],
        ),
    ]
)


def parse_contents(filename):
    try:
        if 'db' in filename:
            # Assume that the user uploaded a db file
            global db_path
            history = h.History("sqlite:///" + db_path)
            all_runs = h.History.all_runs(history)
            list_run_ids = [x.id for x in all_runs]
            # get file name form full path
            name = filename.split('/')[-1]
            # name = filename
            # time = "last modified: " + str(
            #     datetime.datetime.fromtimestamp(date)
            # )
            dist_df = history.get_distribution()
            para_list = []
            for col in dist_df[0].columns:
                para_list.append(col)
    except Exception as e:
        return (
            'There was an error processing this file.',
            [],
            "",
            dbc.Alert(
                [
                    html.H4("Error!", className="alert-heading"),
                    html.P(
                        f"There were an error while loading the database. "
                        f"{str(e)}.",
                    ),
                ],
                id="user_update_alert",
                is_open=True,
                fade=True,
                color="success",
                duration=3000,
                style={'color': '#006400'},
            ),
        )
    return html.Div(
        [
            html.Button(
                "Name: " + name,
                id='btn-nclicks-1',
            ),
            # html.Button(
            #     time,
            #     id='btn-nclicks-2',
            #     style={'margin-left': '10px'},
            # ),
            html.Br(),
            html.Button(
                '# of Runs: ' + str(len(list_run_ids)),
                id='btn-nclicks-3',
                style={'margin-left': '10px'},
            ),
            html.Button(
                '# of parameters: ' + str(len(para_list)),
                id='btn-nclicks-4',
                style={'margin-left': '10px'},
            ),
            html.Div(id='container-button-timestamp'),
        ],
        style={
            'width': '100%',
            'height': '180px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
        },
    )


def prepare_run_detailes(history):
    try:
        dist_df = history.get_distribution()
        para_list = []
        end_time = history.get_abc().end_time
        start_time = history.get_abc().start_time
        id = history.get_abc().id
        dist_string_eps = history.get_abc().distance_function
        dist_val_eps = dist_string_eps.split('"')[1::2]
        dist_name = dist_val_eps[1]
        pop_str_name = history.get_population_strategy()['name']
        if 'nr_calibration_particles' in history.get_population_strategy():
            pop_str_calib = history.get_population_strategy()[
                'nr_calibration_particles'
            ]
        else:
            pop_str_calib = "None"
        if 'nr_samples_per_parameter' in history.get_population_strategy():
            pop_str_sample_per_par = history.get_population_strategy()[
                'nr_samples_per_parameter'
            ]
        else:
            pop_str_sample_per_par = "None"

        pop_str_nr_particles = history.get_population_strategy()[
            'nr_particles'
        ]
        eps_string_eps = history.get_abc().epsilon_function
        eps_val_eps = re.findall(r':(.*?)\,', eps_string_eps)
        eps_val_eps_last = re.findall(r'weighted":(.*?)\}', eps_string_eps)
        eps_name = eps_val_eps[0]
        eps_init_spe = eps_val_eps[1]
        eps_alpha = eps_val_eps[2]
        eps_quant = eps_val_eps[3]
        eps_weited = eps_val_eps_last[0]
        for col in dist_df[0].columns:
            para_list.append(col)
    except Exception as e:
        return html.Div(
            [
                'There was an error processing this file. ' + e,
            ]
        )
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Run's details"),
                    # html.Hr(),  # horizontal line
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H5("General info:"),
                            html.Label("Start time: " + str(start_time)),
                            html.Label("End time: " + str(end_time)),
                            html.Label("Run's ID: " + str(id)),
                        ],
                        style={
                            'display': 'inline-block',
                            'margin-right': '40px',
                            'vertical-align': 'top',
                            'height': '200px',
                            'fontSize': 14,
                            'margin-bottom': '25px',
                            'border-radius': '5px',
                            'padding': '15px',
                            'position': 'relative',
                            'background-color': 'WhiteSmoke',
                            'box-shadow': '2px 2px 2px lightgrey',
                        },
                    ),
                    html.Div(
                        [
                            html.H5("Distance function:"),
                            html.Label("Name: " + str(dist_name)),
                        ],
                        style={
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-right': '40px',
                            'height': '200px',
                            'margin-bottom': '25px',
                            'border-radius': '5px',
                            'padding': '15px',
                            'position': 'relative',
                            'background-color': 'WhiteSmoke',
                            'box-shadow': '2px 2px 2px lightgrey',
                        },
                    ),
                    html.Div(
                        [
                            html.H5("Population strategy:"),
                            html.Label("Name: " + str(pop_str_name)),
                            html.Label(
                                "Number of calibration particles: "
                                + str(pop_str_calib)
                            ),
                            html.Label(
                                "Number of samples per parameter: "
                                + str(pop_str_sample_per_par)
                            ),
                            html.Label(
                                "Number of particles: "
                                + str(pop_str_nr_particles)
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-right': '40px',
                            'height': '200px',
                            'margin-bottom': '25px',
                            'border-radius': '5px',
                            'padding': '15px',
                            'position': 'relative',
                            'background-color': 'WhiteSmoke',
                            'box-shadow': '2px 2px 2px lightgrey',
                        },
                    ),
                    html.Div(
                        [
                            html.H5("Epsilon function:"),
                            html.Label("Name: " + str(eps_name)),
                            html.Label(
                                "Initial epsilon: " + str(eps_init_spe)
                            ),
                            html.Label("Alpha: " + str(eps_alpha)),
                            html.Label(
                                "Quantile multiplier: " + str(eps_quant)
                            ),
                            html.Label("Weighted: " + str(eps_weited)),
                        ],
                        style={
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'height': '200px',
                            'margin-bottom': '25px',
                            'border-radius': '5px',
                            'padding': '15px',
                            'position': 'relative',
                            'background-color': 'WhiteSmoke',
                            'box-shadow': '2px 2px 2px lightgrey',
                        },
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value='tab-1',
                        parent_className='custom-tabs',
                        className='custom-tabs-container',
                        children=[
                            dcc.Tab(
                                id="tab-1",
                                label='Probability density functions',
                                value='tab-pdf',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                            dcc.Tab(
                                id="tab-2",
                                label='Samples',
                                value='tab-samples',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                            dcc.Tab(
                                id="tab-3",
                                label='Particles',
                                value='tab-particles',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                            dcc.Tab(
                                id="tab-4",
                                label='Epsilons',
                                value='tab-epsilons',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                            dcc.Tab(
                                id="tab-5",
                                label='Credible intervals',
                                value='tab-credible',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                            dcc.Tab(
                                id="tab-6",
                                label='Effective sample sizes',
                                value='tab-effective',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                            ),
                        ],
                        vertical=True,
                    ),
                    html.Div(style={'width': '5%'}),
                    html.Div(id='tabs-content'),
                ],
                style={
                    'display': 'Flex',
                },
            ),
        ],
        style={
            'width': '98%',
            'height': '120px',
            'margin': '10px',
        },
    )


@app.callback(
    dash.dependencies.Output("information_grid", "children"),
    [dash.dependencies.Input('ABC_runs', 'value')],
)
def display_info(smc_id):
    global db_path
    try:
        history = h.History("sqlite:///" + db_path, _id=smc_id)
    except Exception:
        return " "
    global para_list
    dist_df = history.get_distribution()
    para_list.clear()
    for col in dist_df[0].columns:
        para_list.append(col)
    return prepare_run_detailes(history)


@app.callback(
    dash.dependencies.Output("hidden-div", "children"),
    [dash.dependencies.Input('download_path', 'value')],
)
def update_download_path(new_download_path):
    global DOWNLOAD_DIR
    DOWNLOAD_DIR = new_download_path
    return [DOWNLOAD_DIR]


# @app.callback(
#     dash.dependencies.Output("alert-fade", "is_open"),
# )
# def error_handler(message):
#     return message


@app.callback(
    [
        Output('output-data-upload', 'children'),
        Output('ABC_runs', 'options'),
        Output('ABC_runs', 'value'),
        dash.dependencies.Output("alert_div", "children"),
    ],
    [
        Input('upload_file', 'n_clicks'),
        dash.dependencies.Input('upload-data', 'value'),
    ],
)
def update_DB_details(btn_click, db_path_new):
    if btn_click > 0:
        try:
            # file_name = "pyABC_server_" + db_path_new
            # save_file(file_name, list_of_contents[0])
            global db_path
            db_path = db_path_new
            history = h.History("sqlite:///" + db_path_new)
        except Exception as e:
            if e.__class__.__name__ == "TypeError":
                return (
                    "",
                    [],
                    "",
                    dbc.Alert(
                        [
                            html.H4(
                                "There were an error while loading the "
                                "database!",
                                className="alert-heading",
                            ),
                        ],
                        id="user_update_alert",
                        is_open=True,
                        fade=True,
                        color="error",
                        duration=2000,
                        style={'color': 'red'},
                    ),
                )
            else:
                return (
                    "",
                    [],
                    "",
                    dbc.Alert(
                        [
                            html.H4("Error!", className="alert-heading"),
                            html.P(
                                f"Please upload a database. " f"{str(e)}.",
                            ),
                        ],
                        id="user_update_alert",
                        is_open=True,
                        fade=True,
                        color="success",
                        duration=5000,
                        style={'color': 'red'},
                    ),
                )
        all_runs = h.History.all_runs(history)
        list_run_ids = [x.id for x in all_runs]
        if all_runs is not None:
            children = [parse_contents(db_path_new)]
            return (
                children,
                [{'label': name, 'value': name} for name in list_run_ids],
                list_run_ids[-1],
                dbc.Alert(
                    [
                        html.H4("Great!", className="alert-heading"),
                        html.P(
                            "The database was loaded successfully.",
                        ),
                    ],
                    id="user_update_alert",
                    is_open=True,
                    fade=True,
                    color="success",
                    duration=3000,
                    style={'color': '#006400'},
                ),
            )
    else:
        return (
            "",
            [],
            "",
            dbc.Alert(
                [
                    html.H4("Error!", className="alert-heading"),
                    html.P(
                        "Please upload a database!.",
                    ),
                ],
                id="user_update_alert",
                is_open=True,
                fade=True,
                color="success",
                duration=5000,
            ),
        )


def prepare_fig_tab(smc_id):
    global db_path
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    dist_df = history.get_distribution()
    para_list = []
    for col in dist_df[0].columns:
        para_list.append(col)
    return html.Div(
        [
            html.Div(
                [
                    html.Label('Select parameter: '),
                    dcc.Dropdown(
                        id="parameters",
                        options=[
                            {'label': name, 'value': name}
                            for name in para_list
                        ],
                        multi=True,
                    ),
                ]
            ),
            html.Br(),
            html.Div(
                [
                    html.Label('Select parameter: '),
                    dcc.Dropdown(
                        id="figure_type",
                        options=[
                            {'label': 'pdf_1d', 'value': 'pdf_1d'},
                            {'label': 'pdf_2d', 'value': 'pdf_2d'},
                            {
                                'label': 'sample_numbers',
                                'value': 'sample_numbers',
                            },
                            {'label': 'epsilons', 'value': 'epsilons'},
                            {
                                'label': 'credible_intervals',
                                'value': 'credible_intervals',
                            },
                            {
                                'label': 'effective_sample_sizes',
                                'value': 'effective_sample_sizes',
                            },
                        ],
                    ),
                ]
            ),
            html.Br(),
            html.Div(
                [
                    # "ABC run plots: ",
                    # html.Br(),
                    # html.Br(),
                    html.Img(
                        id='abc_run_plot',
                        src='data:image/png;base64,{}'.format(square_base64),
                    ),  # img element,
                ]
            ),
        ],
        style={
            'width': '100%',
            'height': '120px',
            'borderRadius': '5px',
            'margin': '30px',
        },
    )


@app.callback(
    dash.dependencies.Output('tabs-content', 'children'),  # src attribute
    [
        dash.dependencies.Input('ABC_runs', 'value'),
        dash.dependencies.Input("tabs", "value"),
    ],
)
def update_figure_ABC_run(smc_id, f_type):
    # create some matplotlib graph
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    global para_list
    if f_type == "tab-pdf":
        return [
            html.Label('Select parameter: '),
            dcc.Dropdown(
                id="parameters",
                options=[{'label': name, 'value': name} for name in para_list],
                multi=True,
                value=[para_list[0]],
            ),
            html.Div(
                [
                    # "ABC run plots: ",
                    html.Img(
                        id='abc_run_plot',
                        src='data:image/png;base64,{}'.format(square_base64),
                    ),  # img element,
                ],
                style={'textAlign': 'center'},
            ),
            html.Div(
                [
                    "x limits: ",
                    html.Br(),
                    html.Br(),
                    html.Div(
                        children=[
                            "min: ",
                            dcc.Input(
                                id="bar_min",
                                placeholder='min',
                                type='number',
                                value='0',
                            ),
                            "max: ",
                            dcc.Input(
                                id="bar_max",
                                placeholder='max',
                                type='number',
                                value='20',
                            ),
                            "Generate code: ",
                            html.Button('Generate', id='copy', n_clicks=0),
                            html.Hr(),
                            dcc.RangeSlider(
                                id='my-range-slider',
                                min=0,
                                max=20,
                                marks=None,
                                value=[0, 0],
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                            ),
                        ]
                    ),
                    html.Div(id='output-container-range-slider'),
                ]
            ),
        ]
    elif f_type == "tab-samples":
        pyabc.visualization.plot_sample_numbers(history)
    elif f_type == "tab-particles":
        fig2, ax2 = plt.subplots()
        particles = (
            history.get_nr_particles_per_population()
            .reset_index()
            .rename(columns={"index": "t", "t": "particles"})
            .query("t >= 0")
        )
        ax2.set_xlabel('population index')
        ax2.set_ylabel('Particles')
        ax2.plot(particles["t"], particles["particles"])

    elif f_type == "tab-epsilons":
        pyabc.visualization.plot_epsilons(history)
    elif f_type == "tab-credible":
        # buf = io.BytesIO()  # in-memory files
        #
        # plt.savefig(buf, format="png")  # save to the above file object
        # data = base64.b64encode(buf.getbuffer()).decode()

        return [
            html.Label('Select parameter: '),
            dcc.Dropdown(
                id="parameters",
                options=[{'label': name, 'value': name} for name in para_list],
                style={'color': 'red'},
                multi=True,
                value=[para_list[0]],
            ),
            html.Div(
                [
                    # "ABC run plots: ",
                    # html.Br(),
                    # html.Br(),
                    html.Img(
                        id='abc_run_plot',
                        src='data:image/png;base64,{}'.format(square_base64),
                    ),
                ],
                style={'textAlign': 'center'},
            ),
            "Generate code: ",
            html.Button('Generate', id='copy', n_clicks=0),
            html.Div(
                children=[
                    "min: ",
                    dcc.Input(
                        id="bar_min",
                        placeholder='min',
                        type='number',
                        value='0',
                    ),
                    "max: ",
                    dcc.Input(
                        id="bar_max",
                        placeholder='max',
                        type='number',
                        value='20',
                    ),
                    html.Hr(),
                    dcc.RangeSlider(
                        id='my-range-slider',
                        min=0,
                        max=20,
                        marks=None,
                        value=[0, 0],
                    ),
                ],
                style={'display': 'none'},
            ),
        ]
    elif f_type == "tab-effective":
        pyabc.visualization.plot_effective_sample_sizes(history)
    elif f_type == "tab-pdf":
        df, w = history.get_distribution(m=0)
        pyabc.visualization.plot_kde_matrix(df, w)
    buf = io.BytesIO()  # in-memory files
    plt.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode(
        "utf8"
    )  # encode to html elements
    # plt.close()
    return [
        # html.Br(),
        # # "ABC run plots: ",
        # html.Br(),
        # html.Br(),
        html.Div(
            [
                html.Img(
                    id='abc_run_plot',
                    src="data:image/png;base64,{}".format(data),
                ),
                # img element,
            ],
            style={'textAlign': 'center'},
        ),
        "Generate code: ",
        html.Button('Generate', id='copy', n_clicks=0),
    ]


@app.callback(
    [
        dash.dependencies.Output('abc_run_plot', 'src'),
        dash.dependencies.Output('bar_min', 'value'),
        dash.dependencies.Output('bar_max', 'value'),
    ],
    [
        dash.dependencies.Input('ABC_runs', 'value'),
        dash.dependencies.Input('parameters', 'value'),
        dash.dependencies.Input("tabs", "value"),
        dash.dependencies.Input('my-range-slider', 'value'),
    ],
)
def update_figure_ABC_run_parameters(smc_id, parameters, f_type, bar_val):
    # create some matplotlib graph
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    buf = None
    global para_list
    if f_type == "tab-pdf":
        fig, ax = plt.subplots()
        if len(parameters) == 1:
            for t in range(history.max_t + 1):
                df, w = history.get_distribution(m=0, t=t)
                if bar_val == [0, 0]:
                    xmin = df[parameters[0]].min()
                    xmax = df[parameters[0]].max()
                else:
                    xmin = bar_val[0]
                    xmax = bar_val[1]

                pyabc.visualization.plot_kde_1d(
                    df,
                    w,
                    xmin=xmin,
                    xmax=xmax,
                    x=parameters[0],
                    ax=ax,
                    label="PDF t={}".format(t),
                )
            ax.legend()
        elif len(parameters) == 2:
            df, w = history.get_distribution(m=0)
            if bar_val == [0, 0]:
                xmin = df[parameters[0]].min()
                xmax = df[parameters[0]].max()
            else:
                xmin = bar_val[0]
                xmax = bar_val[1]

            pyabc.visualization.plot_kde_2d(
                df, w, parameters[0], parameters[1], xmin=xmin, xmax=xmax
            )

        else:
            df, w = history.get_distribution(m=0)
            pyabc.visualization.plot_kde_matrix(df, w)
            if bar_val == [0, 0]:
                xmin = df[parameters[0]].min()
                xmax = df[parameters[0]].max()
            else:
                xmin = bar_val[0]
                xmax = bar_val[1]

    elif f_type == "tab-credible":
        df, w = history.get_distribution()
        if bar_val == [0, 0]:
            xmin = df[parameters[0]].min()
            xmax = df[parameters[0]].max()
        else:
            xmin = bar_val[0]
            xmax = bar_val[1]

        if len(parameters) == 0:
            return
        pyabc.visualization.plot_credible_intervals(
            history,
            levels=[0.95, 0.9, 0.5],
            ts=[0, 1, 2, 3, 4],
            show_mean=True,
            show_kde_max_1d=True,
            par_names=parameters,
        )
    buf = io.BytesIO()  # in-memory files
    plt.gcf().set_size_inches(7, 5)
    plt.tight_layout()
    plt.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode(
        "utf8"
    )  # encode to html elements
    # plt.close()
    return (
        "data:image/png;base64,{}".format(data),
        math.floor(xmin),
        math.ceil(xmax),
    )


# @app.callback(
#     dash.dependencies.Output('output-container-range-slider', 'children'),
#     [dash.dependencies.Input('my-range-slider', 'value')])
# def update_output(value):
#     return 'You have selected "{}"'.format(value)


# # update the bar when a new image in loaded
# @app.callback(
#     [dash.dependencies.Output('my-range-slider', 'value')],
#     [dash.dependencies.Input('abc_run_plot', 'src')])
# def update_output(value):
#     history = h.History("sqlite:///" + db_path, _id=smc_id)
#     df, w = history.get_distribution(m=0, t=t)
#     values = [df[x].min(), df[x].max()]
#     return [0,2]


@app.callback(
    [
        dash.dependencies.Output("my-range-slider", "min"),
        dash.dependencies.Output("my-range-slider", "max"),
    ],
    [
        dash.dependencies.Input("bar_min", "value"),
        dash.dependencies.Input("bar_max", "value"),
    ],
)
def number_render(min, max):
    return int(min), int(max)


def save_file(name, content):
    """Decode and store a file uploaded with Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    with open(os.path.join(DOWNLOAD_DIR, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def save_code_snippet(code_pt2):
    code_pt1 = 'import pyabc\n\
import matplotlib.pyplot as plt\n\
# Please please adapt this: db_path\n\
history = pyabc.storage.History("sqlite:///" + db_path)\n'
    code_pt3 = 'plt.show()'
    code = code_pt1 + code_pt2 + code_pt3
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    with open(os.path.join(DOWNLOAD_DIR, "code_snippet.py"), "w") as fp:
        fp.write(code)


@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('copy', 'n_clicks'),
    Input("tabs", "value"),
)
def displayClick(btn_click, tab_type):
    if btn_click > 0:
        code_pt2 = ""
        if tab_type == 'tab-pdf':
            code_pt2 = (
                '# Please please adapt this: '
                'lower_lim, upper_lim, parameter\n'
                'for t in range(history.max_t + 1):\n'
                '    df, w = history.get_distribution(m=0, t=t)\n'
                '    pyabc.visualization.plot_kde_1d(\n'
                '    df, w, xmin=lower_lim[0], xmax=bar_val[1],\n'
                '    x=parameter, ax=ax,\n'
                '    label="PDF t={}".format(t))\n'
            )
        elif tab_type == 'tab-samples':
            code_pt2 = 'pyabc.visualization.plot_sample_numbers(history)\n'
        elif tab_type == 'tab-particles':
            code_pt2 = (
                'fig2, ax2 = plt.subplots()\n'
                'particles = (history.get_nr_particles_'
                'per_population()'
                '.reset_index().'
                'rename(columns={"index": "t", "t": "particles"})'
                '.query("t >= 0"))\n\
                       ax2.set_xlabel("population index")\n\
                       ax2.set_ylabel("Particles")\n\
                       ax2.plot(particles["t"], particles["particles"])\n'
            )
        elif tab_type == 'tab-epsilons':
            code_pt2 = 'pyabc.visualization.plot_epsilons(history)\n'
        elif tab_type == 'tab-effective':
            code_pt2 = (
                'pyabc.visualization.plot_effective_' 'sample_sizes(history)\n'
            )
        elif tab_type == 'tab-credible':
            code_pt2 = (
                '# Please please adapt this: parameter\n'
                'pyabc.visualization.plot_credible_intervals(\n'
                'history, levels=[0.95, 0.9, 0.5], '
                'ts=[0, 1, 2, 3, 4],\n'
                'how_mean=True, '
                'show_kde_max_1d=True,par_names=parameter)\n'
            )
        save_code_snippet(code_pt2)


@click.command()
@click.option(
    "--debug",
    default=False,
    type=bool,
    help="Whether to run the server in debug mode",
)
@click.option(
    "--port",
    default=8050,
    type=int,
    help="The port on which the server runs (default: 8050)",
)
@click.option(
    "--host",
    default="localhost",
    type=str,
    help="Host name (default: 127.0.0.1 / localhost)",
)
def run_app(host, port, debug):
    app.run(host=host, port=port, debug=debug)
