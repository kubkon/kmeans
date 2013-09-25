# Test dataset
dataset = {
    'trees': {
        'geom': 1,
        'stats': 0,
        'logic': 3,
        'group': 3,
        'grad': 0,
        'disc': 3,
        'real': 2,
        'sup': 3,
        'unsup': 2,
        'multi': 3
    },
    'rules': {
        'geom': 0,
        'stats': 0,
        'logic': 3,
        'group': 3,
        'grad': 1,
        'disc': 3,
        'real': 2,
        'sup': 3,
        'unsup': 0,
        'multi': 2
    },
    'naive_bayes': {
        'geom': 1,
        'stats': 3,
        'logic': 1,
        'group': 3,
        'grad': 1,
        'disc': 3,
        'real': 1,
        'sup': 3,
        'unsup': 0,
        'multi': 3
    },
    'knn': {
        'geom': 3,
        'stats': 1,
        'logic': 0,
        'group': 2,
        'grad': 2,
        'disc': 1,
        'real': 3,
        'sup': 3,
        'unsup': 0,
        'multi': 3
    },
    'linear_classifier': {
        'geom': 3,
        'stats': 0,
        'logic': 0,
        'group': 0,
        'grad': 3,
        'disc': 1,
        'real': 3,
        'sup': 3,
        'unsup': 0,
        'multi': 0
    },
    'linear_regression': {
        'geom': 3,
        'stats': 1,
        'logic': 0,
        'group': 0,
        'grad': 3,
        'disc': 0,
        'real': 3,
        'sup': 3,
        'unsup': 0,
        'multi': 1
    },
    'logistic_regression': {
        'geom': 3,
        'stats': 2,
        'logic': 0,
        'group': 0,
        'grad': 3,
        'disc': 1,
        'real': 3,
        'sup': 3,
        'unsup': 0,
        'multi': 0
    },
    'svm': {
        'geom': 2,
        'stats': 2,
        'logic': 0,
        'group': 0,
        'grad': 3,
        'disc': 2,
        'real': 3,
        'sup': 3,
        'unsup': 0,
        'multi': 0
    },
    'kmeans': {
        'geom': 3,
        'stats': 2,
        'logic': 0,
        'group': 1,
        'grad': 2,
        'disc': 1,
        'real': 3,
        'sup': 0,
        'unsup': 3,
        'multi': 1
    },
    'gmm': {
        'geom': 1,
        'stats': 3,
        'logic': 0,
        'group': 0,
        'grad': 3,
        'disc': 1,
        'real': 3,
        'sup': 0,
        'unsup': 3,
        'multi': 1
    },
    'associations': {
        'geom': 0,
        'stats': 0,
        'logic': 3,
        'group': 3,
        'grad': 0,
        'disc': 3,
        'real': 1,
        'sup': 0,
        'unsup': 3,
        'multi': 1
    }
}

row_headers = ['trees', 'rules', 'naive_bayes',
               'knn', 'linear_classifier', 'linear_regression',
               'logistic_regression', 'svm', 'kmeans',
               'gmm', 'associations']

column_headers = ['geom', 'stats', 'logic', 'group',
                  'grad', 'disc', 'real', 'sup',
                  'unsup', 'multi']

def to_2d_array():
    # Transform dataset into a 2D array
    dataset_arr = []
    for row, i in zip(row_headers, range(len(row_headers))):
        dataset_arr.append([])
        for col in column_headers:
            dataset_arr[i].append(dataset[row][col])

    return dataset_arr
