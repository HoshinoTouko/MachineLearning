'''
@File: SklearnDefaultDecisionTree.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2017, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Create at: 2018-03-28 14:41
@Desc: 
'''
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO

import pydot


def predict(feature_train, target_train, feature_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(feature_train, target_train)
    predict_results = dt_model.predict(feature_test)

    dot_data = StringIO()
    export_graphviz(dt_model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("iris.pdf")

    return predict_results
