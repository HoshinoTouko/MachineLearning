'''
@File: Test.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2017, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Create at: 2018-03-28 14:31
@Desc: 
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from IrisDecisionTree import SklearnDefaultDecisionTree
from IrisDecisionTree.HandwriteDecisionTree import build_tree, predict, predict_all

iris = datasets.load_iris()

iris_feature = iris.data
iris_target = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

'''
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class: 
  -- Iris Setosa
  -- Iris Versicolour
  -- Iris Virginica
'''

# Split train and test dataset
feature_train, feature_test, target_train, target_test = train_test_split(
    iris_feature, iris_target, test_size=0.3, random_state=42
)

if __name__ == '__main__':
    # Scikit-learn default decision tree.
    predict_results_skl = SklearnDefaultDecisionTree.predict(
        feature_train, target_train, feature_test
    )
    scores_skl = accuracy_score(predict_results_skl, target_test)
    print('SklearnDefaultDecisionTree.predict')
    print(scores_skl)

    root_node = build_tree(
        feature_train, target_train, 1
    )
    predict_results_hdw = predict_all(root_node, feature_test)
    scores_hdw = accuracy_score(predict_results_hdw, target_test)
    print('HandwriteDecisionTree.predict')
    print(scores_hdw)
