.. _web_visualization:

Visualization and analysis
==========================

Plotting
--------

pyABC offers a variety of routines to visualize and analyze results.
Usage examples are given in the :ref:`Examples section<examples>`.
An overview of all plotting routines can be found in the
:ref:`Visualization API documentation <api_visualization>`.
Further, functions to correctly extract statistics, in particular
on importance samples obtained via ABCSMC, can be found in the
:ref:`Weighted statistics API documentation <api_weighted_statistics>`.

Visualization server
--------------------

pyABC compes with a visualization server based on the data visualization
tool Dash,
to interactively display information on running and completed jobs.

You can launch it from the command line with

.. code-block:: bash

    abc-server-dash

It opens per default a web server on port 8050.

You should see something similar to the following:

.. image:: server_screenshots/dash_main.png


A copy of the selected database will be saved on ``/tmp/``, if you want to change the path, you can enter the full path of the new folder in the text box.

.. image:: server_screenshots/dash_path.png

To upload your database, you can click or drag and drop your file in the dashed rectangle. Please note that the database file should have the extension ``.db``

The time taken to upload the database to the server will depend on the database size. Once you select your database, you will notice that the title bar of the tab changed to ``uploading``. Please wait until the uploading process finishes.

.. image:: server_screenshots/dash_update.png

If the metadata of your file appears, that means your database file was uploaded successfully.

.. image:: server_screenshots/dash_meta.png

you can then select the requested run's ID.

Once you slelct the run's ID, the left side of the page will be updated to show more details about the selected run.

You can then select one of the plots by selecting one of the tabs under ``Run's plots``. For some of the tabs, you will be asked to select one parameter, or more, from the dropdown list.

To save any plot, right-click on the plot and then select ``save image as``.


To get help, type in the command line:

.. code-block:: bash

   abc-server-dash --help

To get more information on available options, such as selecting another port:

Visualization server -- using Flask
-----------------------------------

Note: This web-server is deprecated and no longer maintained.

Further, pyABC comes with a web server based on the micro web framework Flask.
You can launch it from the command line with

.. code-block:: bash

    abc-server-flask <databasename>

It opens per default a web server on port 5000.

You should see something similar to the following:

.. image:: server_screenshots/main.png

Via "Go to ABC Run List", you can see all running and finished ABC runs, which you can then inspect in more detail.

You can get overviews over the models:

.. image:: server_screenshots/model_overview.png

Information about individual model parameters for each model and time point is also displayed:

.. image:: server_screenshots/model_detail.png

Type in the command line

.. code-block:: bash

   abc-server-flask --help

To get more information on available options, such as selecting another port:

.. code-block:: bash

   abc-server-flask --port=8888 <databasename>
