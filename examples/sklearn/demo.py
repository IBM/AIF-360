import sys
sys.path.append('../..')
from aif360.sklearn.datasets import fetch_adult, fetch_compas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
# from aif360.metrics import statistical_parity_difference
import pandas as pd

dataset = "compas"
if(dataset == "adult"):
    X, y, sample_weight = fetch_adult(numeric_only=True)
elif(dataset == "compas"):
    X, y = fetch_compas(numeric_only=True, binary_race=True)
else:
    raise ValueError("Unknown dataset")


X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

y = pd.Series(y.factorize(sort=True)[0], index=y.index)


scaler = MinMaxScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
print(X)



(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)

clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))



from aif360.sklearn.explainers.bias_explainer import explain_statistical_parity, explain_equalized_odds, explain_predictive_parity, draw_plot

if(True):
    result, fairXplainer = explain_statistical_parity(clf, X_train, 
                sensitive_features=['race'], maxorder=1, 
                verbose=True)


    plt = draw_plot(result, 
                    draw_waterfall=True, 
                    fontsize=22, 
                    labelsize=18, 
                    figure_size=(10, 6), 
                    title="Exact statistical parity: {}".format(round(fairXplainer.statistical_parity_sample(), 3)), 
                    xlim=None,
                    x_label="Influence on statistical parity", 
                    text_x_pad=0.02, 
                    text_y_pad=0.1, 
                    result_x_pad=0.02, 
                    result_y_location=0.5, 
                    delete_zero_weights=False
        )
    # plt.show()
    plt.clf()

if(False):
    results, fairXplainers = explain_equalized_odds(clf, X_train, y_train, sensitive_features=['race'], maxorder=1, verbose=True)
    print(results)
    print(fairXplainers)

    index_max_unfairness = 0
    if(fairXplainers[0].statistical_parity_sample() < fairXplainers[1].statistical_parity_sample()):
        index_max_unfairness = 1

    plt = draw_plot(results[index_max_unfairness], 
                    draw_waterfall=True, 
                    fontsize=22, 
                    labelsize=18, 
                    figure_size=(10, 6), 
                    title="Exact equalized odds: {}".format(round(fairXplainers[index_max_unfairness].statistical_parity_sample(), 3)), 
                    xlim=None,
                    x_label="Influence on equalized odds", 
                    text_x_pad=0.02, 
                    text_y_pad=0.1, 
                    result_x_pad=0.02, 
                    result_y_location=0.5, 
                    delete_zero_weights=False
    )
    # plt.show()
    plt.clf()

if(False):
    results, fairXplainers = explain_predictive_parity(clf, X_train, y_train, sensitive_features=['race'], maxorder=1, verbose=True)
    print(results)
    print(fairXplainers)

    index_max_unfairness = 0
    if(fairXplainers[0].statistical_parity_sample() < fairXplainers[1].statistical_parity_sample()):
        index_max_unfairness = 1

    plt = draw_plot(results[index_max_unfairness],
                    draw_waterfall=True,
                    fontsize=22,
                    labelsize=18,
                    figure_size=(10, 6),
                    title="Exact predictive parity: {}".format(round(fairXplainers[index_max_unfairness].statistical_parity_sample(), 3)),
                    xlim=None,
                    x_label="Influence on predictive parity",
                    text_x_pad=0.02,
                    text_y_pad=0.1,
                    result_x_pad=0.02,
                    result_y_location=0.5,
                    delete_zero_weights=False
    )

    plt.show()
    plt.clf()

