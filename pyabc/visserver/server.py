import os
import json
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import click
from pyabc import History
import pandas as pd
import bokeh.plotting.helpers as helpers
from bokeh.plotting import figure
# this has to be set before the other bokeh imports
helpers.DEFAULT_PALETTE = ['#000000',   # Wong nature colorblind palette
                           '#e69f00',
                           '#56b4e9',
                           '#009e73',
                           '#f0e442',
                           '#0072b2',
                           '#d55e00',
                           '#cc79a7']
from bokeh.embed import components  # noqa: E402
from bokeh.resources import INLINE  # noqa: E402
from bokeh.models.widgets import Panel, Tabs  # noqa: E402

BOKEH = INLINE


class PlotScriptDiv:
    def __init__(self, script, div):
        self.script = script
        self.div = div


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def main():
    return render_template("index.html")


@app.route("/abc")
def abc_overview():
    history = app.config["HISTORY"]
    runs = history.all_runs()
    return render_template("abc_overview.html", runs=runs)


class ABCInfo:
    def __init__(self, abc):
        self.abc = abc

    def __getattr__(self, item):
        json_str = getattr(self.abc, item).replace("'", '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}


@app.route("/abc/<int:abc_id>")
def abc_detail(abc_id):
    history = app.config["HISTORY"]
    history.id = abc_id
    abc = ABCInfo(history.get_abc())
    model_probabilities = history.get_model_probabilities()
    model_ids = model_probabilities.columns
    model_probabilities.columns = list(map("{}".format,
                                           model_probabilities.columns))
    model_probabilities = model_probabilities.reset_index()
    if len(model_probabilities) > 0:
        populations = history.get_all_populations()
        populations = populations[populations.t >= 0]
        particles = (history.get_nr_particles_per_population().reset_index()
                     .rename(columns={"index": "t", "t": "particles"})
                     .query("t >= 0"))

        melted = pd.melt(model_probabilities, id_vars="t", var_name="m",
                         value_name="p")
        melted = melted.convert_objects()
        melted["m"] = pd.to_numeric(melted["m"])

        # although it might seem cumbersome, not using the bkcharts
        # package works more reliably

        prob_plot = figure()
        prob_plot.xaxis.axis_label = 'Generation t'
        prob_plot.yaxis.axis_label = 'Probability'
        for c, (m, data) in zip(helpers.DEFAULT_PALETTE, melted.groupby("m")):
            prob_plot.line(data["t"], data["p"],
                           legend="Model " + str(m), color=c,
                           line_width=2)

        particles_fig = figure()
        particles_fig.xaxis.axis_label = 'Generation t'
        particles_fig.yaxis.axis_label = 'Particles'
        particles_fig.line(particles["t"], particles["particles"],
                           line_width=2)

        samples_fig = figure()
        samples_fig.xaxis.axis_label = 'Generation t'
        samples_fig.yaxis.axis_label = 'Samples'
        samples_fig.line(populations["t"], populations["samples"],
                         line_width=2)

        eps_fig = figure()
        eps_fig.xaxis.axis_label = 'Generation t'
        eps_fig.yaxis.axis_label = 'Epsilon'
        eps_fig.line(populations["t"], populations["epsilon"],
                     line_width=2)

        plot = Tabs(tabs=[
            Panel(child=prob_plot, title="Probability"),
            Panel(child=samples_fig, title="Samples"),
            Panel(child=particles_fig, title="Particles"),
            Panel(child=eps_fig, title="Epsilon")])
        plot = PlotScriptDiv(*components(plot))

        return render_template("abc_detail.html",
                               abc_id=abc_id,
                               plot=plot,
                               BOKEH=BOKEH,
                               model_ids=model_ids,
                               abc=abc)
    return render_template("abc_detail.html",
                           abc_id=abc_id,
                           plot=PlotScriptDiv("", "Exception: No data found."),
                           BOKEH=BOKEH,
                           abc=abc)


@app.route("/abc/<int:abc_id>/model/<int:model_id>/t/<t>")
def abc_model(abc_id, model_id, t):
    history = app.config["HISTORY"]
    history.id = abc_id
    if t == "max":
        t = history.max_t
    else:
        t = int(t)
    df, w = history.get_distribution(model_id, t)
    df["CDF"] = w
    tabs = []

    model_ids = history.get_model_probabilities().columns
    for parameter in [col for col in df if col != "CDF"]:
        plot_df = df[["CDF", parameter]].sort_values(parameter)
        plot_df_cumsum = plot_df.cumsum()
        plot_df_cumsum[parameter] = plot_df[parameter]
        f = figure()
        f.line(x=plot_df_cumsum[parameter], y=plot_df_cumsum["CDF"])
        p = Panel(child=f, title=parameter)
        tabs.append(p)
    if len(tabs) == 0:
        plot = PlotScriptDiv("", "This model has no Parameters")
    else:
        plot = PlotScriptDiv(*components(Tabs(tabs=tabs)))
    return render_template("model.html",
                           abc_id=abc_id,
                           model_id=model_id,
                           plot=plot,
                           BOKEH=BOKEH,
                           model_ids=model_ids,
                           t=t,
                           available_t=list(range(history.max_t+1)))


@app.route("/info")
def server_info():
    history = app.config["HISTORY"]
    return render_template("server_info.html", db_path=history.db_file(),
                           db_size=round(history.db_size, 2))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@click.command()
@click.option("--debug", default=False, type=bool,
              help="Whether to run the server in debug mode")
@click.option("--port", default=5000, type=int,
              help="The port on which the server runs")
@click.argument("db")
def run_app(db, debug, port):
    db = os.path.expanduser(db)
    history = History("sqlite:///" + db)
    app.config["HISTORY"] = history
    app.run(debug=debug, port=port)
