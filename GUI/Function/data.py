from sklearn.preprocessing import StandardScaler
def preProcess(data):
    if 'start' or 'end' in data.columns:
        data = data.drop(columns=['start', 'end'])
    if 'video' in data.columns:
        data = data.drop(columns=['video'])
    if 'color' in data.columns:
        data = data.drop(columns=['color'])

    y = labels = data['target']
    features = data.drop(columns=['target'])
    X = features = StandardScaler().fit_transform(features)
    return X, y
def preProcess_transfer(data):
    y = labels = data['target']
    features = data.drop(columns=['target'])
    X = features = StandardScaler().fit_transform(features)
    return X, y

def get_video(video):

    if video in [0,3,6,9]:
        return '0403'
    elif video in [1, 4, 7, 10]:
        return '0608'
    elif video in [12, 15, 18, 21]:
        return '0713'
    elif video in [13, 16, 19, 22]:
        return '0720'
def get_point(video):
    import datetime
    a, b, c, d = "0403.wmv", "0608.wmv", '0713.wmv', "0720.wmv"
    v_dict = {a: '11:23:49.986', b: '10:18:19.591',
              c: '12:12:45.550', d: '13:05:56.705'}
    if video in a:
        video = a
    if video in b:
        video = b
    if video in c:
        video =c
    if video in d:
        video =d
    time = datetime.datetime.strptime(v_dict[video], "%H:%M:%S.%f")
    deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                   microseconds=time.microsecond)
    return deltatime

def get_color(video_num):
    if '1' in video_num:
        return 'green'
    if '2' in video_num:
        return 'kuro'
    if '3' in video_num:
        return 'orange'
    if '4' in video_num:
        return 'pink'
    if '5' in video_num:
        return 'green'
    if '6' in video_num:
        return 'kuro'
    if '7' in video_num:
        return 'orange'
    if '8' in video_num:
        return 'pink'