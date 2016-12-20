from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import sys
from bokeh.charts import Line, Histogram
from bokeh.embed import components
import os
from pyabc import History
from collections import namedtuple
from bokeh.resources import CDN
BOKEH = CDN

PlotScriptDiv = namedtuple("PlotScriptDiv", "script div")

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def main():
    return render_template("index.html")


@app.route("/abc")
def abc_overview():
    runs = history.all_runs()
    return render_template("abc_overview.html", runs=runs)


@app.route("/abc/<int:abc_id>")
def abc_detail(abc_id):
    history.id = abc_id
    model_probabilities = history.get_model_probabilities()
    model_ids = model_probabilities.columns
    model_probabilities.columns = list(map(lambda x: "Model {}".format(x),
                                           model_probabilities.columns))
    if len(model_probabilities) > 0:
        plot = Line(model_probabilities)
        plot = PlotScriptDiv(*components(plot))
        return render_template("abc_detail.html",
                               abc_id=abc_id,
                               plot=plot,
                               BOKEH=BOKEH,
                               model_ids=model_ids)
    return render_template("abc_detail.html",
                           abc_id=abc_id,
                           plot=PlotScriptDiv("", "Exception: No data found."),
                           BOKEH=BOKEH)


@app.route("/abc/<int:abc_id>/model/<int:model_id>")
def abc_model(abc_id, model_id):
    history.id = abc_id
    max_t = history.max_t
    df, w = history.weighted_parameters_dataframe(max_t, model_id)
    df["CDF"] = w
    plots = []
    for parameter in [col for col in df if col != "CDF"]:
        plot_df = df[["CDF", parameter]].sort_values(parameter)
        plot_df_cumsum = plot_df.cumsum()
        plot_df_cumsum = plot_df_cumsum.set_index("CDF")
        plots.append(PlotScriptDiv(*components(Line(plot_df_cumsum))))
    return render_template("model.html", abc_id=abc_id, model_id=model_id)


if __name__ == '__main__':
    db = os.path.expanduser(sys.argv[1])
    history = History("sqlite:///" + db)
    app.run(debug=True)
