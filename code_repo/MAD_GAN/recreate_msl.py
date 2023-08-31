import os
import numpy as np
from torch.utils.data import DataLoader


def load_dataset(dataset):
    folder = 'processed/SMAP'
    loader = []

    label = os.path.join(folder, dataset[0])
    test = os.path.join(folder, dataset[1])
    train = os.path.join(folder, dataset[2])

    loader.append(np.load(label))
    loader.append(np.load(test))
    loader.append(np.load(train))

    train_loader = loader[2]
    test_loader = loader[1]
    labels = loader[0]*1
    # msl = os.path.join('processed', dataset)
    # msl = np.load(os.path.join(smd, 'P-10_test.npy'))
    return train_loader, test_loader, labels


if __name__ == '__main__':
    msl_file_list = sorted(os.listdir('processed/SMAP'))

    train_loader, test_loader, label_loader = np.ndarray([]), np.ndarray([]), np.ndarray([])

    for i in range(0, len(msl_file_list), 3):
        train, test, label = load_dataset(msl_file_list[i:i+3])

        if i == 0:
            train_loader = train
            test_loader = test
            label_loader = label
        else:
            train_loader = np.vstack((train_loader, train))
            test_loader = np.vstack((test_loader, test))
            label_loader = np.vstack((label_loader, label))

    # np.save('processed1/SMAP/SMAP_test.npy', test_loader)
    # np.save('processed1/SMAP/SMAP_test_label.npy', label_loader)
    # np.save('processed1/SMAP/SMAP_train.npy', train_loader)

    print(train_loader.shape)
    print(test_loader.shape)
    print(label_loader.shape)