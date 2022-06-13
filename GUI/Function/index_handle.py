from alipy.index import IndexCollection
from alipy.experiment import State
from alipy.data_manipulate import split
from sklearn.preprocessing import StandardScaler

def cancel_step(select_ind, lab, unlab):
    lab = IndexCollection(lab)
    unlab = IndexCollection(unlab)
    unlab.update(select_ind)
    lab.difference_update(select_ind)
    lab_list, unlab_list = [], []
    for i in lab:
        lab_list.append(i)
    for i in unlab:
        unlab_list.append(i)
    return lab_list, unlab_list


def update(select_ind, lab, unlab):
    lab = IndexCollection(lab)
    unlab = IndexCollection(unlab)
    lab.update(select_ind)
    unlab.difference_update(select_ind)
    lab_list, unlab_list = [],[]
    for i in  lab:
        lab_list.append(i)
    for i in  unlab:
        unlab_list.append(i)
    return lab_list, unlab_list

def save_state(data, select_ind, current_ac):
    quried_label = data.loc[select_ind,['target']]
    st = State(select_ind,current_ac,queried_label=quried_label)
    return st
def separate(data):
    if 'start' or 'end' in data.columns:
        data = data.drop(columns=['start', 'end'])
    if 'video' in data.columns:
        data = data.drop(columns=['video'])
    if 'color' in data.columns:
        data = data.drop(columns=['color'])

    y = labels = data['target']
    features = data.drop(columns=['target'])

    X = features = StandardScaler().fit_transform(features)

    train, test, lab, unlab = split(X, y, test_ratio=0.3, initial_label_rate=0.2, split_count=1, all_class=True,
                                    saving_path='.')
    train_list, test_list, lab_list, unlab_list = [] , [] ,[] ,[]
    for i in train[0]:
        train_list.append(i)
    for i in test[0]:
        test_list.append(i)
    for i in  lab[0]:
        lab_list.append(i)
    for i in  unlab[0]:
        unlab_list.append(i)
    return X[lab_list], y[lab_list]