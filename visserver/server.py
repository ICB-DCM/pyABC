from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import sys
from bokeh.charts import Line
from bokeh.embed import components
import os
from pyabc import History
from bokeh.resources import INLINE
BOKEH = INLINE


class PlotScriptDiv:
    def __init__(self, script, div):
        self.script=script
        self.div = div


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


@app.route("/abc/<int:abc_id>/model/<int:model_id>/t/<t>")
def abc_model(abc_id, model_id, t):
    history.id = abc_id
    if t == "max":
        t = history.max_t
    else:
        t = int(t)
    df, w = history.weighted_parameters_dataframe(t, model_id)
    df["CDF"] = w
    plots = []
    for parameter in [col for col in df if col != "CDF"]:
        plot_df = df[["CDF", parameter]].sort_values(parameter)
        plot_df_cumsum = plot_df.cumsum()
        plot_df_cumsum[parameter] = plot_df[parameter]
        print(plot_df_cumsum)
        p = PlotScriptDiv(*components(Line(x=parameter, y="CDF", data=plot_df_cumsum)))
        p.parameter_name = parameter
        plots.append(p)
    return render_template("model.html",
                           abc_id=abc_id,
                           model_id=model_id,
                           plots=plots,
                           BOKEH=BOKEH,
                           t=t,
                           available_t=list(range(history.max_t)))


if __name__ == '__main__':
    db = os.path.expanduser(sys.argv[1])
    history = History("sqlite:///" + db)
    app.run(debug=True)
