import gzip
import numpy as np
import os
import pickle
import urllib.request

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.join('datasets')
os.makedirs(dataset_dir, exist_ok=True)
save_file = os.path.join(dataset_dir, 'mnist.pkl')

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _change_one_hot_label(X):
    """
    OneHotEncodingする

    [a, b, c] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    # 今回0~9なので10列
    T = np.zeros((X.size, 10))
    # 該当する個所のみ1にする
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def _convert_numpy():
    """
    画像とラベルをnumpy配列としてロードする
    """
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def _download(file_name):
    """
    指定されたファイルをローカルにダウンロードする
    """
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        return

    print(f'Downloading {file_name} ...')
    urllib.request.urlretrieve(os.path.join(url_base, file_name), file_path)
    print('Done.')


def _download_mnist():
    """
    各ファイルをダウンロードする
    """
    for v in key_file.values():
        _download(v)


def _init_mnist():
    """
    mnistデータのPickleファイルを作成する
    """
    _download_mnist()
    dataset = _convert_numpy()
    print(f'Creating pickle...')
    with open(save_file, 'wb') as f:
        # pickle.dump(obj, file, protocol)
        # protocol=-1はHIGHEST_PROTOCOL
        pickle.dump(dataset, f, -1)
    print('Done.')


def _load_img(file_name):
    """
    画像をnumpy配列にする
    """
    file_path = os.path.join(dataset_dir, file_name)
    print(f'Converting {file_name} to NumPy Array...')
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print('Done.')

    return data


def _load_label(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print(f'Converting {file_name} to NumPy Array...')
    with gzip.open(file_path, 'rb') as f:
        # numpy.frombuffer(buffer, dtype, offset)
        # 先頭8byteをスキップする
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print('Done.')

    return labels


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """[summary]
    mnistを読み込む関数。外部から参照される

    Keyword Arguments:
        normalize {bool} -- 正規化するか (default: {True})
        flatten {bool} -- 一次元配列にするか (default: {True})
        one_hot_label {bool} -- one-hot配列にするか (default: {False})

    Returns:
        訓練画像, 訓練ラベル, テスト画像, テストラベル
    """
    if not os.path.exists(save_file):
        _init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ['train_img', 'test_img']:
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] = dataset[key] / 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ['train_img', 'test_img']:
            # reshapeメソッドで-1は他の次元から推測される
            # shapeは(データ数, 1列, 28x28の画像データ)となる
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']
