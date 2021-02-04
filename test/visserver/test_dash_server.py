import dash
import dash_html_components as html


def test_one(dash_duo):
    app = import_app("dash_test.app")
    dash_duo.start_server(app)
    dash_duo.wait_for_text_to_equal("h1", "Hello Dash", timeout=4)
