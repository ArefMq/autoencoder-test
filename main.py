#!/usr/bin/env python2

# requirements:
# - scikit-neuralnetwork
# - sklearn
# - pandas


from dataset import load_regression_dataset, FEATURE_LIST
from sklearn import decomposition
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

from util import R2, print_errors

colors = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.',
          'b*', 'g*', 'r*', 'c*', 'm*', 'y*', 'k*']


def plot_pca(x, y, title, img_id):
    if x.shape[1] < 2:
        return

    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x_2d = pca.transform(x)

    for _x, _y in zip(x_2d, y):
        plt.plot(_x[0], _x[1], colors[_y])

    title = ', '.join([s.split(' ')[0] for s in title])
    plt.title(title)
    # plt.show()
    plt.savefig('%s.png' % img_id)
    plt.clf()
    plt.cla()


def get_feature_list(counter):
    selected = []
    valid_features = {i: 2 ** i for i in range(len(FEATURE_LIST))}
    for f, v in valid_features.items():
        if v & counter:
            selected.append(f)
    return [FEATURE_LIST[t] for t in selected]


def feature_selection():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

    x_train, x_test, y_train, y_test = load_regression_dataset()
    config_table = {}

    for counter in range(1, 2 ** len(FEATURE_LIST)):
        subset_features = get_feature_list(counter)
        subset_x = x_train[subset_features]

        # print counter, '=>', subset_features
        # plot_pca(subset_x, y_train, subset_features, 'img_%d' % counter)

        q = QDA()
        q.fit(subset_x, y_train)

        subset_test_x = x_test[subset_features]
        y_pred = q.predict(subset_test_x)

        # Process
        a = y_pred.flatten()
        b = y_test
        precision, recal, f_score, support = precision_recall_fscore_support(a, b, average='macro')

        r2 = R2(y_pred, y_test.values)
        if r2 > 0:
            config_table[tuple(subset_features)] = (r2, f_score, counter)

    iter = 0
    for key, value in sorted(config_table.iteritems(), key=lambda (k, v): (v[1], k), reverse=True):
        print '- %s\n(R2 = %0.4f | f-score=%0.4f | id=%d)' % ('\n- '.join(key), value[0], value[1], value[2])
        print '-----------------------------------------------------------------------'

        if iter < 10:
            iter += 1
        else:
            break


def auto_encode(x, y):
    from sknn import ae, mlp

    # Initialize auto-encoder for unsupervised learning.
    myae = ae.AutoEncoder(
        layers=[
            ae.Layer("Tanh", units=8),
            ae.Layer("Sigmoid", units=4)],
        learning_rate=0.002,
        n_iter=10)

    # Layerwise pre-training using only the input data.
    myae.fit(x)

    # Initialize the multi-layer perceptron with same base layers.
    mymlp = mlp.Regressor(
        layers=[
            mlp.Layer("Tanh", units=8),
            mlp.Layer("Sigmoid", units=4),
            mlp.Layer("Linear")])

    # Transfer the weights from the auto-encoder.
    myae.transfer(mymlp)
    # Now perform supervised-learning as usual.
    mymlp.fit(x, y)
    return mymlp


def run_auto_encode():
    x_train, x_test, y_train, y_test = load_regression_dataset()
    reg = auto_encode(x_train, y_train)

    # y_pred = reg.predict(x_test)
    # a = y_pred.flatten()
    # b = y_test
    # precision, recal, f_score, support = precision_recall_fscore_support(a, b, average='macro')
    print_errors(reg, x_train, y_train, x_test, y_test, prf=True)



def main():
    # feature_selection()
    run_auto_encode()



if __name__ == "__main__":
    main()
