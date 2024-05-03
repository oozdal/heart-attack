import plotly, os
import plotly.express as px
import numpy as np
import pandas as pd
import pathlib
from plotly.offline import init_notebook_mode
from sklearn.metrics import confusion_matrix as cm_sklearn

plotly.io.renderers.default = "browser"
init_notebook_mode(connected=True) # initiate notebook for offline plot


def _plotly_write_json(plotly_fig, filename='plot.json', pretty=True):
    """
    Convert a Plotly figure to JSON and write it to a file or writeable object
    
    Parameters
    ----------
    plotly_fig: plotly.graph_objs._figure.Figure
                Figure from plotly
    filename: str, default='plot.json'
                Output JSON filename
    pretty: (bool (default True)) – True if JSON representation should be pretty-printed, False if representation should be as compact as possible.

    Returns
    -------
    plotly_fig.write_json(filename)
    """

    return plotly.io.write_json(plotly_fig, filename, pretty)

def confusion_matrix_plot(clf, y_test, y_pred, classes=None, save_fig_path=None):
    """
    Plot a confusion matrix for multiclass using MLextend functionalities. At some point we could merge this function with confusion_matrix_plot() (binary).

    Parameters
    ----------
    clf : estimator instance (either sklearn.Pipeline, imblearn.Pipeline or a classifier)
        PRE-FITTED classifier or a PRE-FITTED Pipeline in which the last estimator is a classifier.
    y_test : array
        Known labels
    y_pred : array
        Predictions
    classes : list or array
        classes names
    save_fig_path : str, default=None
        Full path where to save the plot. Will generate the folders if they don't exist already. HTML is allowed.

    Returns
    -------
    plotly_fig: plotly.graph_objs._figure.Figure
        Figure from plotly
    """

    cm = cm_sklearn(y_true = y_test, y_pred = y_pred, normalize='all') # normalized confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3) # restrict up to 3 decimal points

    # if classes is not given, set it to np.unique(y_test)
    if classes == None:
        classes_predefined = False
        df = pd.DataFrame(cm)
        classes = list(np.unique(y_test))[::-1]
        classes = [str(element) for element in classes]
    else:
        classes_predefined = True
        df = pd.DataFrame(np.flip(cm).T)

    # class numbers should be integer unless string is given
    df['Index'] = classes[::-1] # Reversing a list

    try:
        df['Index'] = df['Index'].astype(float)
        df['Index'] = df['Index'].astype(int)
        df.set_index('Index', inplace=True)
    except:
        df.set_index('Index', inplace=True)

    cm = df.to_numpy() #DataFrame to numpy array
    sorted_classes = df.index.to_list()

    if clf.__class__.__name__ == 'Pipeline':
        classifier_name = clf['clf'].__class__.__name__
    else:
        classifier_name = clf.__class__.__name__

    # Plotly figure
    plotly_fig = px.imshow(df, text_auto=True,
                           labels=dict(x="Actual Values", y="Predicted Values", color="metric scale"),
                           color_continuous_scale='YlGn')

    yaxis_dict = dict(
        tickmode='array',
        dtick=20,
        tickvals=None,
        ticktext=sorted_classes,
        showticklabels=True,
        color='black'
    )

    if (len(np.unique(y_test).tolist()) == 2): yaxis_dict["tickvals"] = np.arange(len(sorted_classes))
    elif classes_predefined == False: yaxis_dict["tickvals"] = np.arange(1, len(sorted_classes)+1)
    else: yaxis_dict["tickvals"] = np.arange(len(sorted_classes))

    layout = dict(
        autosize=False,
        xaxis=dict(
            tickmode='array',
            dtick=20,
            tickvals=np.arange(len(sorted_classes)),
            ticktext=sorted_classes,
            showticklabels=True,
            color='black'
        ),
        yaxis = yaxis_dict
    )

    plotly_fig.update_layout(layout)
    plotly_fig.update_xaxes(title_font=dict(size=18))
    plotly_fig.update_yaxes(title_font=dict(size=18))

    if (len(np.unique(y_test).tolist()) == 2):
        plotly_fig.update_layout(
            title_text=classifier_name + ' Confusion Matrix',
            title_font_color="white")
    else:
        plotly_fig.update_layout(
            title_text=classifier_name + ' Multiclass Confusion Matrix',
            title_font_color="white")

    if save_fig_path != None and save_fig_path.split('.')[-1] == 'html':
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plotly.offline.plot(plotly_fig, filename=save_fig_path, auto_open=False) # Saving as an HTML file
        pre, ext = os.path.splitext(save_fig_path)
        _plotly_write_json(plotly_fig, filename=os.path.abspath(pre) + '.json') # Saving as a JSON file

    return plotly_fig


def read_from_json(json_file):
    return plotly.io.read_json(json_file)