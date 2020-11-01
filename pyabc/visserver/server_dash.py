import datetime
from dash.dependencies import State
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pyabc.storage import history as h
import pyabc
import matplotlib.pyplot as plt
import io
import base64
import re
import os
import click
import pathlib

static = str(pathlib.Path(__file__).parent.absolute())+'/assets/'
DOWNLOAD_DIRECTORY = "/tmp/"
db_path = DOWNLOAD_DIRECTORY
parameter = ""
square_png = static+'square.png'
square_base64 = base64.b64encode(open(square_png, 'rb').read()).decode('ascii')
pyABC_png = static+'pyABC_logo.png'
pyABC_base64 = base64.b64encode(open(pyABC_png, 'rb').read()).decode('ascii')
para_list = []
colors = {
    'div': '#595959',
}
# Initialise the app
app = dash.Dash(__name__)

# Define the app
app.layout = app.layout = html.Div(children=[
    html.Div(className='row',  # Define the row element
             children=[
                 # header
                 html.Div(className='four columns div-user-controls',
                          children=[
                              html.Img(id='pyABC_logo',
                                       src='data:image/png;base64,{}'.format(
                                           pyABC_base64)), html.Br(),
                              html.H1(
                                  'pyABC: Likelihood free inference'),
                              html.H3('(Web server)'),
                              html.Hr(),
                              html.Div(["A copy of the db will be save in: ",
                                        dcc.Input(id="download_path",
                                                  placeholder='/tmp/',
                                                  type='text',
                                                  value='/tmp/')]),
                              html.Div(id="hidden-div",
                                       style={"display": "none"}),
                              dcc.Upload(
                                  id='upload-data',
                                  children=html.Div([
                                      'Drag and Drop or ',
                                      html.A('Select Files')
                                  ]),
                                  style={
                                      'width': '100%',
                                      'height': '60px',
                                      'lineHeight': '60px',
                                      'borderWidth': '1px',
                                      'borderStyle': 'dashed',
                                      'borderRadius': '5px',
                                      'textAlign': 'center',
                                      'margin': '10px'}, multiple=True),
                              # show details in boxes once db is loaded
                              html.Div([
                                  html.Div(id='output-data-upload'),
                                  html.Div([
                                      html.Div(["ABC runs: ",
                                                dcc.Dropdown(id='ABC_runs', ),
                                                ])], style={'width': '100%',
                                                            'height': '100%'}),
                              ]),
                          ]
                          ),  # Define the left element
                 html.Div(id='information_grid',
                          className='eight columns div-for-charts bg-grey',
                          style={'height': '1200px'})
                 # Define the right element
             ])
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    try:
        if 'db' in filename:
            # Assume that the user uploaded a db file
            global db_path
            history = h.History("sqlite:///" + db_path)
            all_runs = h.History.all_runs(history)
            list_run_ids = [x.id for x in all_runs]
            name = filename
            time = "last modified: " + str(
                datetime.datetime.fromtimestamp(date))
            dist_df = history.get_distribution()
            para_list = []
            for col in dist_df[0].columns:
                para_list.append(col)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.Button("Name: " + name,
                    id='btn-nclicks-1',
                    ),
        html.Button(time,
                    id='btn-nclicks-2', style={'margin-left': '10px'}
                    ),
        html.Br(),
        html.Button('# of Runs: ' + str(len(list_run_ids)),
                    id='btn-nclicks-3', style={'margin-left': '10px'},
                    ),
        html.Button('# of parameters: ' + str(len(para_list)),
                    id='btn-nclicks-4', style={'margin-left': '10px'},
                    ),
        html.Div(
            id='container-button-timestamp')
    ], style={
        'width': '100%',
        'height': '180px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'solid',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',

    }, )


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
                'nr_calibration_particles']
        else:
            pop_str_calib = "None"
        if 'nr_samples_per_parameter' in history.get_population_strategy():
            pop_str_sample_per_par = history.get_population_strategy()[
                'nr_samples_per_parameter']
        else:
            pop_str_sample_per_par = "None"

        pop_str_nr_particles = history.get_population_strategy()[
            'nr_particles']
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
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.Div([
            html.H1('Runs details'),
            html.Hr(),  # horizontal line
        ]),
        html.Div([
            html.Div([
                html.H6("General info:"),
                html.Label("Start time: " + str(start_time)),
                html.Label("End time: " + str(end_time)),
                html.Label("Run's ID: " + str(id)),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'height': '200px', 'fontSize': 14}),
            html.Div([
                html.H5("Distance function:"),
                html.Label("Name: " + str(dist_name)),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'margin-left': '40px', 'height': '200px', }),
            html.Div([
                html.H2("Population strategy:"),
                html.Label("Name: " + str(pop_str_name)),
                html.Label(
                    "Number of calibration particles: " + str(pop_str_calib)),
                html.Label(
                    "Number of samples per parameter: " + str(
                        pop_str_sample_per_par)),
                html.Label(
                    "Number of particles: " + str(pop_str_nr_particles)),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'margin-left': '40px', 'height': '200px', }),
            html.Div([
                html.H1("Epsilon function:"),
                html.Label("Name: " + str(eps_name)),
                html.Label("Initial epsilon: " + str(eps_init_spe)),
                html.Label("Alpha: " + str(eps_alpha)),
                html.Label("Quantile multiplier: " + str(eps_quant)),
                html.Label("Weighted: " + str(eps_weited)),
            ], style={'display': 'inline-block', 'vertical-align': 'top',
                      'margin-left': '40px', 'height': '200px', })
        ]),
        html.Div([
            html.H1('Runs plots'),
            html.Hr(), html.Br(),
            # horizontal line
        ]),
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
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            id="tab-2",
                            label='Samples',
                            value='tab-samples',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            id="tab-3",
                            label='Particles',
                            value='tab-particles',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            id="tab-4",
                            label='Epsilons',
                            value='tab-epsilons',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            id="tab-5",
                            label='Credible intervals',
                            value='tab-credible',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            id="tab-6",
                            label='Effective sample sizes',
                            value='tab-effective',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        )

                    ]),
                html.Div(id='tabs-content')
            ]),

    ], style={
        'width': '98%',
        'height': '120px',
        'margin': '10px',
    })


@app.callback(
    dash.dependencies.Output("information_grid", "children"),
    [dash.dependencies.Input('ABC_runs', 'value')]
)
def display_info(smc_id):
    global db_path
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    global para_list
    dist_df = history.get_distribution()
    para_list.clear()
    for col in dist_df[0].columns:
        para_list.append(col)
    return prepare_run_detailes(history)


@app.callback(
    dash.dependencies.Output("hidden-div", "children"),
    [dash.dependencies.Input('download_path', 'value')]
)
def update_download_path(new_download_path):
    global DOWNLOAD_DIRECTORY
    DOWNLOAD_DIRECTORY = new_download_path
    return [DOWNLOAD_DIRECTORY]


@app.callback([Output('output-data-upload', 'children'),
               Output('ABC_runs', 'options')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_DB_details(list_of_contents, list_of_names, list_of_dates):
    file_name = "pyABC_server_" + list_of_names[0]
    save_file(file_name, list_of_contents[0])
    global db_path
    db_path = DOWNLOAD_DIRECTORY + file_name
    print(db_path)
    history = h.History("sqlite:///" + db_path)
    all_runs = h.History.all_runs(history)
    list_run_ids = [x.id for x in all_runs]
    print(list_run_ids)
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children, [{'label': name, 'value': name}
                          for name in list_run_ids]


def prepare_fig_tab(smc_id):
    global db_path
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    dist_df = history.get_distribution()
    para_list = []
    for col in dist_df[0].columns:
        para_list.append(col)
    return html.Div([
        html.Div([
            html.Label('Select parameter: '),
            dcc.Dropdown(
                id="parameters",
                options=[{'label': name, 'value': name} for name in para_list],
                multi=True
            )]), html.Br(),
        html.Div([
            html.Label('Select parameter: '),
            dcc.Dropdown(
                id="figure_type", options=[
                    {'label': 'pdf_1d', 'value': 'pdf_1d'},
                    {'label': 'pdf_2d', 'value': 'pdf_2d'},
                    {'label': 'sample_numbers', 'value': 'sample_numbers'},
                    {'label': 'epsilons', 'value': 'epsilons'},
                    {'label': 'credible_intervals',
                     'value': 'credible_intervals'},
                    {'label': 'effective_sample_sizes',
                     'value': 'effective_sample_sizes'},
                ]
            )]), html.Br(),

        html.Div(["ABC run plots: ", html.Br(), html.Br(),
                  html.Img(id='abc_run_plot',
                           src='data:image/png;base64,{}'.format(
                               square_base64)),  # img element,
                  ]),
    ], style={
        'width': '100%',
        'height': '120px',
        'borderRadius': '5px',
        'margin': '30px',
    }, )


@app.callback(
    dash.dependencies.Output('tabs-content', 'children'),  # src attribute
    [dash.dependencies.Input('ABC_runs', 'value'),
     dash.dependencies.Input("tabs", "value")]
)
def update_figure_ABC_run(smc_id, f_type):
    # create some matplotlib graph
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    global para_list
    if f_type == "tab-pdf":
        return [html.Label('Select parameter: '),
                dcc.Dropdown(
                    id="parameters",
                    options=[{'label': name, 'value': name} for name in
                             para_list], multi=True
                ), html.Div(["ABC run plots: ", html.Br(), html.Br(),
                             html.Img(id='abc_run_plot',
                                      src='data:image/png;base64,{}'.format(
                                          square_base64)),  # img element,
                             ], style={'textAlign': 'center'})]
    elif f_type == "tab-samples":
        pyabc.visualization.plot_sample_numbers(history)
    elif f_type == "tab-particles":
        fig2, ax2 = plt.subplots()
        particles = (history.get_nr_particles_per_population().reset_index()
                     .rename(columns={"index": "t", "t": "particles"})
                     .query("t >= 0"))
        ax2.set_xlabel('population index')
        ax2.set_ylabel('Particles')
        ax2.plot(particles["t"], particles["particles"])

    elif f_type == "tab-epsilons":
        pyabc.visualization.plot_epsilons(history)
    elif f_type == "tab-credible":
        return [html.Label('Select parameter: '),
                dcc.Dropdown(
                    id="parameters",
                    options=[{'label': name, 'value': name} for name in
                             para_list], multi=True
                ), html.Div(["ABC run plots: ", html.Br(), html.Br(),
                             html.Img(id='abc_run_plot',
                                      src='data:image/png;base64,{}'.format(
                                          square_base64)),  # img element,
                             ], style={'textAlign': 'center'})]
    elif f_type == "tab-effective":
        pyabc.visualization.plot_effective_sample_sizes(history)
    elif f_type == "tab-pdf":
        df, w = history.get_distribution(m=0)
        pyabc.visualization.plot_kde_matrix(df, w)
    buf = io.BytesIO()  # in-memory files

    plt.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode(
        "utf8")  # encode to html elements
    plt.close()
    return [html.Br(), "ABC run plots: ", html.Br(), html.Br(), html.Div([
         html.Img(
             id='abc_run_plot',
             src="data:image/png;base64,{}".format(
                 data)),
         # img element,
         ],
     style={
         'textAlign': 'center'})]


@app.callback(
    dash.dependencies.Output('abc_run_plot', 'src'),  # src attribute
    [dash.dependencies.Input('ABC_runs', 'value'),
     dash.dependencies.Input('parameters', 'value'),
     dash.dependencies.Input("tabs", "value")]
)
def update_figure_ABC_run_parameters(smc_id, parameters, f_type):
    # create some matplotlib graph
    history = h.History("sqlite:///" + db_path, _id=smc_id)
    global para_list
    if f_type == "tab-pdf":

        fig, ax = plt.subplots()
        if len(parameters) == 1:
            for t in range(history.max_t + 1):
                df, w = history.get_distribution(m=0, t=t)
                pyabc.visualization.plot_kde_1d(
                    df, w,
                    xmin=0, xmax=5,
                    x=parameters[0], ax=ax,
                    label="PDF t={}".format(t))
            ax.legend()
        elif len(parameters) == 2:
            df, w = history.get_distribution(m=0, )
            pyabc.visualization.plot_kde_2d(df, w, parameters[0],
                                            parameters[1])
        else:
            df, w = history.get_distribution(m=0, )
            pyabc.visualization.plot_kde_matrix(df, w)

    elif f_type == "tab-credible":
        if len(parameters) == 0:
            return
        pyabc.visualization.plot_credible_intervals(
            history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
            show_mean=True, show_kde_max_1d=True, par_names=parameters)
    buf = io.BytesIO()  # in-memory files
    plt.savefig(buf, format="png")  # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode(
        "utf8")  # encode to html elements
    plt.close()

    return "data:image/png;base64,{}".format(data)


def save_file(name, content):
    """Decode and store a file uploaded with Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)
    with open(os.path.join(DOWNLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


@click.command()
@click.option("--debug", default=False, type=bool,
              help="Whether to run the server in debug mode")
@click.option("--port", default=8050, type=int,
              help="The port on which the server runs")
def run_app(debug, port):
    app.run_server(debug=debug, port=port)
