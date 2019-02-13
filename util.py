import random
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc


def generate_random_name():
    name = '%05x' % random.getrandbits(32)
    return 'M-' + name[:5]


def print_initial_message(task, verbose_level=0):
    def __print_message(task, verbose_level):
        print('Running uNEAT-T1 (%s)' % task)

        if verbose_level < 1:
            return

        print('micro-(Neuro-Evolution of Augmenting Topology) Framework -- Version T1')
        print('(C) 2018 - Aref Moqadam Mehr - Under MIT Licenced')
        print('For more information visit: http://github.com/arefmq')

        if verbose_level < 2:
            return
        print('''
Usage:
    - train.py : for training a network and store the result in
                 'result.network.json' file.
    - test.py  : for validating the trained network.
''')

    __print_message(task, verbose_level)
    print('-------------------------------------------------------')


def RSS(a, b):
    return np.sum((a - b) ** 2)


def TSS(a, b):
    y_mean = np.mean(b, 0)
    return np.sum((b - y_mean) ** 2)


def R2(a, b):
    return 1.0 - (RSS(a, b) / TSS(a, b))


def print_errors(reg, x_train, y_train, x_test=None, y_test=None, msg=None, prf=False):
    def _print_detail(msg, a, b, prf):
        a = a.flatten()
        b = b.flatten()

        print '%s error:' % msg
        print ' - RSS = %.3f' % (RSS(a, b))
        print ' - TSS = %.3f' % (TSS(a, b))
        print ' - R^2 = %.3f' % (R2(a, b))
        print ''

        # try:
        if prf:
            a = [0 if i < 0.5 else 1 for i in a]
            b = [0 if i < 0.5 else 1 for i in b]
            precision, recal, f_score, support = precision_recall_fscore_support(a, b, average='macro')
            print ' - Precision: %0.3f\n - Recal: %0.3f\n - F-Score: %0.3f' % (precision, recal, f_score)
            print '---------------------------------------------'
        # except:
        #     pass

    print '\n=============================================\n'
    if msg:
        print msg
    _print_detail('Train', reg.predict(x_train), y_train, prf)
    if x_test is None or y_test is None:
        return

    y_pred = reg.predict(x_test)
    _print_detail('Test', y_pred, y_test, prf)

    return R2(y_pred, y_test)
