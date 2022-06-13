import tkinter as tk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
import datetime
import av
import os
from tkinter.messagebox import showwarning
from sklearn.ensemble import RandomForestClassifier
import warnings
from alipy.experiment import StateIO
from sklearn.metrics import accuracy_score
import time
warnings.filterwarnings('ignore')
from Function.index_handle import cancel_step, update, save_state
from Function.metrics import initial_show,ac_entropy,accuracy_show,entropy_show
from Function.data import get_video, get_point,get_color


class app(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        # 开始时皮卡丘封面
        self.cover = self.cover_1()
        # 定义滚动条.
        self.scale_value = tk.IntVar()
        self.sm = 'move'

        # 定义成员，好进行数据存储
        self.movie,self.vw,self.fp,self.video_path,self.R,self.readyFrame, self.final_time = None,None,None,None,None, None, None
        self.input, self.stream = None, None
        self.time_speed = 1/(33*1.6)
        self.train, self.test, self.lab, self.index, self.unlab, self.entropy,self.in_ac, self.current_ac, self.saver, self.data,self.test_data =None, None, None, [], None, None, None, None, StateIO, None, None
        self.X,self.y = None, None
        self.x_lab, self.y_lab = None, None

        # 这次定义 6/13
        self.X_lab, self.y_lab = None, None
        self.udata, self.ldata = None, None
        self.matrix = None
        self.query = None
        # 这次定义 6/13


        self.dict={0:"移動",1:" 摂食",2:"飲水",3:"羽繕い",4:"身震い",5:"頭かき",6:"尾振り",7:"巣箱に乗る",8:"巣箱を降りる",
                                       9:"止まり木に乗る",10:"止まり木を降りる",11:"静止",12:"休息",13:"砂浴び",14:"探査",15:"首振り",16:"バランス",17:"センサつつき",18:"伸び ",19:"嘴とぎ",20:"地面つつき",
                                       21:"きょろきょろ",22:"つつき攻撃",23:"巣箱つつき",24:"つつかれ",25:"センサつつかれ"}

        self.times= 0
        self.counts,self.accuracys = [],[]
        self.video1 = None
        self.model = RandomForestClassifier(random_state=43)
        # self.in_ac, b = initial_show(self.model,self.data)
        # self.saver = StateIO(0, self.train, self.test, self.lab, self.unlab, initial_point=self.in_ac)
        self.toprank = 800

        self.point = 0
        self.process = pd.DataFrame(columns=["index","target","video","start","end",])
        self.p = None

        # 0是播放状态，1是暂停状态
        self.flag = 0
        self.pause_frame = None
        self.list = None

        # 菜单区域
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='open video', command=self.open_video)
        filemenu.add_command(label='add video_path', command=self.add_video_path)
        filemenu.add_command(label='Initialize', command=self.initial)
        filemenu.add_command(label='Unlabel_pool', command=self.upool)
        filemenu.add_command(label='refresh', command = self.restart)

        # Test columns
        testmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Test', menu=testmenu)
        testmenu.add_command(label='add test dataset', command=self.op_sensor1)

        # Speed columns
        speedmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='speed', menu=speedmenu)
        speedmenu.add_command(label='X 0.5', command=lambda: self.speed(0.58))
        speedmenu.add_command(label='X 0.75', command=lambda: self.speed(0.8))
        speedmenu.add_command(label='X 0.9', command=lambda: self.speed(0.9))
        speedmenu.add_command(label='X 1.0', command= lambda:self.speed(1.6))
        speedmenu.add_command(label='X 1.3', command=lambda: self.speed(3.6))
        speedmenu.add_command(label='X 3', command=lambda: self.speed(0))


        # Transfer columns
        transfermenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Transfer', menu=transfermenu)
        transfermenu.add_command(label='transfer the knowldege', command=self.op_sensor2)

        root.config(menu= menubar)
        # 视频区域
        self.frame0 = tk.Frame(root, width='300', height='380')
        self.frame0.grid(row=0, column=0, sticky='N')
        self.video = tk.Canvas(self.frame0, width='640', height='480', bg='yellow')
        self.video.grid(row=0, column=0)
        # 分隔区域
        self.sep = tk.Label(root, width='1', height='28', bg='#333333')
        self.sep.grid(row=0, column=2)
        # 注册信息区域
        self.frame = tk.Frame(root, width='300', height='380')
        self.frame.grid(row=0, column=3, sticky='N')
        # 右边结果区域
        self.frame2 = tk.Frame(root, width='300', height='380')
        self.frame2.grid(row=0, column=4, sticky='N')

        self.video_name = tk.Label(self.frame, text='Video')
        self.entry1 = tk.Entry(self.frame, width='30')

        import tkinter.ttk
        self.target_name = tk.Label(self.frame, text='Target')
        self.entry2 = tk.ttk.Combobox(self.frame,values=["0 :移動Move","1 : 摂食Eat","2 :飲水Drink","3 :羽繕いPreening","4 :身震いShivering","5 :頭かきHead scratch","6 :尾振りTail swing","7 :巣箱に乗るGet on the nest  box","8 :巣箱を降りるGet off the nest box",
                                       "9 :止まり木に乗る Get on the perch","10:止まり木を降りる Get off the perch","11:静止 Stop","12:休息 Rest","13:砂浴びDust bathing","14:探査Litter exploration","15:首振りHead swing","16:バランスTo keep balance","17:センサつつきPeck the sensor","18:伸びStretching","19:嘴とぎBeak sharpening","20:地面つつきPeck the ground",
                                       "21:きょろきょろLook around","22:つつき攻撃Attack another hens","23:巣箱つつきPeck the nest box","24:つつかれPecked","25:センサつつかれPecked the sensor"], width='28')

        self.color = tk.Label(self.frame, text='Color')
        self.entry_X = tk.Entry(self.frame,width='30')
        self.start_time = tk.Label(self.frame, text='Start_Time')
        self.entry3 = tk.Entry(self.frame, width='30')

        self.end_time = tk.Label(self.frame, text='End_time')
        self.entry4 = tk.Entry(self.frame, width='30')

        self.video_name.grid(row=0, column=0, sticky='N')
        self.entry1.grid(row=1, column=0, sticky='N')
        self.target_name.grid(row=2, column=0, sticky='N')
        self.entry2.grid(row=3, column=0, sticky='N')

        # Feb 13
        self.color.grid(row=4, column=0, sticky='N')
        self.entry_X.grid(row=5,column=0, sticky='N')

        self.start_time.grid(row=6, column=0, sticky='N')
        self.entry3.grid(row=7, column=0, sticky='N')
        self.end_time.grid(row=8, column=0, sticky='N')
        self.entry4.grid(row=9, column=0)
        self.input_list = tk.Button(self.frame, text='zip', command=lambda: self.zip1)
        # self.input_list.grid(row=8, column=0, sticky='N')

        # self.get_index  = tk.Button(self.frame, text='query', command= lambda: self.print_index())
        # self.get_index.grid(row=9, column=0, sticky='N')

        self.start_label = tk.Button(self.frame0, text='skip', command= self.skip)
        self.start_label.grid(row=2, column=0, sticky='E')

        self.trainning = tk.Button(self.frame, text='label', command=self.training)
        self.trainning.grid(row=11, column=0, sticky='W')

        self.cancel = tk.Button(self.frame, text='cancel', command=self.cancel)
        self.cancel.grid(row=11, column=0, sticky='E')

        # text区域
        self.text = tk.Text(self.frame2, width='50', height='40')
        self.text.grid(row=0, column=0)
        # 保存打包成文件
        self.save = tk.Button(self.frame2, text='save', command=self.form_data)
        self.save.grid(row=1, column=0, sticky='NE')
        self.show_entropy = tk.Button(self.frame2, text='entropy', command=self.show_entropy)
        self.show_entropy.grid(row=1, column=0, sticky='N')

        self.query_top_1 = tk.Button(self.frame2, text='query', command = self.query1)
        self.query_top_1.grid(row=1,column=0, sticky='NW')
        self.training_data = None

        # 进度条, command = self.set_video
        self.Tscale = tk.Scale(self.frame0, orient='horizonta', variable=self.scale_value, length='635')
        self.Tscale.grid(row=1, column=0, sticky='N')
        # 单击绑定事件
        self.Tscale.bind('<Button-1>', self.scale_state1)
        # 松开触发事件
        self.Tscale.bind('<ButtonRelease-1>', self.scale_state2)

        # self.scale_adjust = tk.Scale(root, orient = 'horizonta', length='635', variable= self.scale_value, command= self.set_video)
        # self.scale_adjust.grid(row=2, column=0, sticky='N')
        # 播放暂停按钮
        self.play_btn = tk.Button(self.frame0, text='play', command=self.judge)
        self.play_btn.grid(row=2, column=0, sticky='N')
        # 剪辑按钮
        self.count = 1
        # self.clip1 = tk.Button(root, text='start', command=lambda: self.bt_count('start11'))
        # self.clip1.grid(row=3, column=0, sticky='E')
        # self.clip2 = tk.Button(root, text='end', command=lambda: self.bt_count('end22'))
        # self.clip2.grid(row=4, column=0, sticky='E')

        # 打开文件
        self.adjust = tk.Button(self.frame0, text='-4s', command=lambda : self.back_video(2))
        self.adjust.grid(row=2, column=0, sticky='NW')

    def speed(self, speed):
        if speed == 0:
            self.time_speed = 0
        else:
            self.time_speed =1/(speed*33)

    def cancel(self):

        print("交换前的各个数据集的长度,",len(self.lab),len(self.unlab))
        self.lab, self.unlab = cancel_step(self.index[-1], self.lab, self.unlab)
        self.index.pop()
        print("交换后的各个数据集的长度,", len(self.lab), len(self.unlab))
        self.current_ac, self.entropy = ac_entropy(self.X[self.lab, :], self.y[self.lab], self.unlab, self.test,
                                                   self.model, self.data, self.test_data)
        self.accuracys.pop()
        if self.accuracys == None:
            self.accuracys = []
        self.counts.pop()
        if self.counts == None:
            self.counts = []
        print("过去的精度是:",self.current_ac)
        print("\n", self.index)

        print("减少之前的Process 数据集", self.process)
        # 减少process 里面的数据集
        self.process = self.process[0:-1]
        print("减少之后的Process 数据集",self.process)

    def training(self):
        print("chose state:",self.entry2.current())

        if self.entry2.current() == -1:
            tk.messagebox.showwarning("NULL!!!","NULL!!!")
        print(self.X_lab)
        self.X_lab = np.vstack((self.X_lab,self.query))
        print(self.X_lab)
        print(self.y_lab)

        if 'ndarray' not in str(type(self.y_lab)):
            self.y_lab =self.y_lab.values
        self.y_lab = np.hstack((self.y_lab,self.entry2.current()))
        print(self.y_lab)
        self.model.fit(self.X_lab, self.y_lab)




    def AC_graph(self,accuracys, begin,counts):
        from matplotlib import pyplot as plt
        plt.ion()
        fig  = plt.subplot()
        fig.set_title('AL process')
        fig.set_ylabel('Accuracy')
        fig.set_xticks(np.arange(0,counts[-1],1))
        fig.set_yticks(np.arange(0, 1, 0.01))
        fig.plot(counts, accuracys)
        plt.pause(0.5)
        plt.show()


    def skip(self):
        video_string = self.entry1.get()
        total_video =  self.video_path
        self.flag = 1
        for i in total_video:
            if str(video_string) in str(i):
                self.fp = i
                self.movie = cv2.VideoCapture(i)
        self.input = av.open(self.fp)
        self.stream = self.input.streams.video[0]
        # 配置进度条和配置
        start = datetime.datetime.strptime(self.entry3.get(), '%H:%M:%S.%f').time()
        self.times = start.hour * 60 * 60 + start.second + start.minute * 60 + (start.microsecond/1000000)

        self.times = self.times - get_point(self.video1).total_seconds()
        end = datetime.datetime.strptime(self.entry4.get(), '%H:%M:%S.%f').time()
        self.end = end.hour * 60 * 60 + end.second + end.minute * 60
        self.end = self.end - get_point(self.video1).total_seconds()

        self.Tscale.configure(to=int(self.stream.duration / 1000), from_=0)
        # 跳转视频
        wanted_time = self.times * 1000

        self.input.seek(offset=int(self.times * 1000+500), stream=self.stream, any_frame=False)

        while int(self.scale_value.get())*1000 < int((self.end) * 1000) :
            self.play()
        self.play()


        # 满足条件达成循环


    # 取出排名第一的熵值函数
    def query1(self):

        self.start=self.udata.head(1)['start'].values[0]
        self.end = self.udata.head(1)['end'].values[0]
        self.video1 = get_video(self.udata.head(1)['video'].values)
        self.udata = self.udata.iloc[1:,:]

        self.entry1.delete(0,'end')
        self.entry1.insert('end',self.video1)
        self.entry_X.delete(0,'end')
        self.entry_X.insert('end',str(self.udata.head(1)['color'].values[0]))
        self.entry3.delete(0, 'end')
        self.entry3.insert('end', self.start)
        self.entry4.delete(0, 'end')
        self.entry4.insert('end', self.end)
        start = datetime.datetime.strptime(self.entry3.get(), '%H:%M:%S.%f').time()
        self.times = start.hour * 60 * 60 + start.second + start.minute * 60
        self.times = self.times - get_point(self.video1).total_seconds()
        end = datetime.datetime.strptime(self.entry4.get(), '%H:%M:%S.%f').time()
        self.end = end.hour * 60 * 60 + end.second + end.minute * 60
        self.end = self.end - get_point(self.video1).total_seconds()
        # 缩短一行

        self.matrix = self.matrix.iloc[1:,:]
        query_x, query_y = self.preProcess(self.matrix)
        query_x = query_x[:,1:][0]
        self.text.delete('1.0', 'end')
        self.query = query_x
        # 显示提示,为最近的几个标签
        tips = self.model.predict_proba(query_x.reshape(1,-1))
        #print(tips[-1])
        from alipy.query_strategy import QueryInstanceUncertainty
        ent = QueryInstanceUncertainty.calc_entropy(tips)
        print(tips,"\n并且对应的检验熵值是:", ent)
        # 编制 dataframe 提示表格
        actions = ["移動Move                     ","摂食Eat                      ","飲水Drink                    ","羽繕いPreening                ","身震いShivering               ","頭かきHead scratch            ","尾振りTail swing              ","巣箱に乗るGet on the nest  box  ","巣箱を降りるGet off the nest box ",
                                       "止まり木に乗る Get on the perch   ","止まり木を降りる Get off the perch ","静止 Stop                    ","休息 Rest                    ","砂浴びDust bathing            ","探査Litter exploration       ","首振りHead swing              ","バランスTo keep balance        ","センサつつきPeck the sensor      ","伸びStretching               ","嘴とぎBeak sharpening         ","地面つつきPeck the ground       ",
                                       "きょろきょろLook around          ","つつき攻撃Attack another hens  ","巣箱つつきPeck the nest box     ","つつかれPecked                 ","センサつつかれPecked the sensor   ", 'unknow']
        tips = pd.DataFrame(data={"prob":tips[0]}, index=self.model.classes_)
        self.text.insert('end', str(tips))
        tips = tips.sort_values(by='prob',ascending=False)

        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        self.text.insert('end', str(tips))
        # self.text.insert('end', str(self.entropy.to_string(index=False,header=True)[:245]) + '\n')

    # 显示熵值于右侧窗口中

    def show_entropy(self):
        # 需要进行初始化
        self.entropy, self.matrix = entropy_show(self.model,self.X_lab,self.y_lab, self.X_unlab, self.udata )
        self.text.delete('1.0', 'end')
        self.text.insert('end', self.entropy)


    # 菜单栏中重新开始
    def restart(self):
        # 定义成员，好进行数据存储
        self.movie, self.vw, self.fp, self.video_path, self.R, self.readyFrame, self.final_time = None, None, None, None, None, None, None
        self.input, self.stream = None, None
        self.time_speed = 1 / (33 * 1.6)
        self.train, self.test, self.lab, self.index, self.unlab, self.entropy, self.in_ac, self.current_ac, self.saver, self.data, self.test_data = None, None, None, [], None, None, None, None, StateIO, None, None
        self.X, self.y = None, None
        self.dict = {0: "移動", 1: " 摂食", 2: "飲水", 3: "羽繕い", 4: "身震い", 5: "頭かき", 6: "尾振り", 7: "巣箱に乗る", 8: "巣箱を降りる",
                     9: "止まり木に乗る", 10: "止まり木を降りる", 11: "静止", 12: "休息", 13: "砂浴び", 14: "探査", 15: "首振り", 16: "バランス",
                     17: "センサつつき", 18: "伸び ", 19: "嘴とぎ", 20: "地面つつき",
                     21: "きょろきょろ", 22: "つつき攻撃", 23: "巣箱つつき", 24: "つつかれ", 25: "センサつつかれ"}
        self.times = 0
        self.counts, self.accuracys = [], []
        self.video1 = None
        self.model = RandomForestClassifier(random_state=43)
        # self.in_ac, b = initial_show(self.model,self.data)
        # self.saver = StateIO(0, self.train, self.test, self.lab, self.unlab, initial_point=self.in_ac)
        self.toprank = 800
        self.point = 0
        self.process = None
        self.p = None
        # 0是播放状态，1是暂停状态
        self.flag = 0
        self.pause_frame = None
        self.list = None

    def back_video(self, times):
        self.input.seek(offset=int((self.scale_value.get()-2) * 1000), stream=self.stream, any_frame=False)
        #videocapture.set(0,value)
    def scale_state1(self, event):
        self.flag=1
    def scale_state2(self, event):
        self.point = self.scale_value.get()
        # self.movie.set(0, (self.point) * 1000)
        self.input.seek(offset=int(self.point * 1000), stream=self.stream, any_frame=False)

        self.play()
        #print(self.scale_value.get())

    def cover_1(self):
        tImage = Image.open("./design/cover.jpg")
        cover = ImageTk.PhotoImage(tImage)
        return cover

    def open_video(self):
        # 找到文件路径
        self.fp = filedialog.askopenfilename()
        self.input = av.open(self.fp)
        self.stream = self.input.streams.video[0]

        # 配置进度条
        self.Tscale.configure(to=int(self.stream.duration/1000), from_=0)
        while self.flag == 0:
            self.input.seek(offset=int(0 * 1000), stream=self.stream, any_frame=False)
            self.play()


    def add_video_path(self):
        self.video_path = filedialog.askopenfilenames()

    def initial(self):
        self.fp_sensors = filedialog.askopenfilenames()
        list = []
        for i in self.fp_sensors:
            video_num = i[-6:-4]
            sample = pd.read_csv(i)
            sample['color'] = get_color(video_num)
            list.append(sample)
        # 打开时初始化数据
        self.ldata =  pd.concat(list, ignore_index= True)
        from Function.index_handle import separate
        self.X_lab, self.y_lab = separate(self.ldata)


        # 初始化对象, 标注池， 以后新标签将会添加进这两个里面
        print(self.X_lab,'\n', self.y_lab)

    def upool(self):
        self.fp_sensors = filedialog.askopenfilenames()
        list = []
        for i in self.fp_sensors:
            video_num = i[-6:-4]
            sample = pd.read_csv(i)
            sample['color'] = get_color(video_num)
            list.append(sample)
        # 打开时初始化数据
        self.udata = pd.concat(list, ignore_index=True)
        self.X_unlab, self.y_unlab = self.preProcess(self.udata)

    def preProcess(self, data):
        from sklearn.preprocessing import StandardScaler
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



    def op_sensor1(self):
        self.fp_sensors_test = filedialog.askopenfilenames()
        list = []
        for i in self.fp_sensors_test:
            video_num = i[-6:-4]
            sample = pd.read_csv(i)
            sample['color'] = get_color(video_num)
            list.append(sample)
        self.test_data = pd.concat(list, ignore_index=True)

    def op_sensor2(self):
        p = filedialog.askopenfilename()
        knowledge = pd.read_csv(p)
        #print(knowledge)
        from Function.data import preProcess_transfer,preProcess
        X,y = preProcess_transfer(knowledge)
        X_test,y_test = preProcess(self.test_data)
        self.model.fit(X,y)
        ac = accuracy_score(y_true=y_test, y_pred=self.model.predict(X_test))
        #print("transfer result:",ac)

    # 用于暂停
    def judge(self):
        if self.flag == 0:
            self.flag = 1
            print("进入暂停状态")
            self.pause()
        elif self.flag == 1:
            self.flag = 0
            print("进入播放状态")
            # only flag=0 ,play the video, otherwise, yellow label.
            while self.flag == 0:
                self.play()

    # 用于暂停
    def pause(self):
        self.video.create_image(0, 0, anchor='nw', image=self.pause_frame)

    def play(self):
        # seek
            for i in self.input.decode(self.stream):
                # to PIL.image
                score = i.pts
                i.reformat(width=640, height=480)
                i = i.to_image()
                img2 = ImageTk.PhotoImage(i)
                self.pause_frame = img2
                self.video.create_image(0, 0, anchor='nw', image=img2)
                self.Tscale.set(int(score / 1000))
                self.video.update()

                time.sleep(self.time_speed)
                if self.flag==1:
                    break



    # 用于某视频的初始时刻
    def delta_time(self):
        # 判断是哪个视频
        a, b, c, d = "0403.wmv", "0608.wmv", '0713.wmv', "0720.wmv"
        v_dict = {a: '11:23:49.986', b: '10:18:19.591',
                  c: '12:12:45.550', d: '13:05:56.705'}

        if a in self.fp:
            self.list = [0,3,6,9]
            time =  datetime.datetime.strptime(v_dict[a], "%H:%M:%S.%f")
            deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                           microseconds=time.microsecond)
            return deltatime
        elif b in self.fp:
            self.list = [1, 4, 7, 10]
            time =  datetime.datetime.strptime(v_dict[b], "%H:%M:%S.%f")
            deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                           microseconds=time.microsecond)
            return deltatime
        elif c in self.fp:
            self.list = [12, 15, 18, 21]
            time =  datetime.datetime.strptime(v_dict[c], "%H:%M:%S.%f")
            deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                           microseconds=time.microsecond)
            return deltatime
        elif d in self.fp:
            self.list = [13, 16, 19, 22]
            time =  datetime.datetime.strptime(v_dict[d], "%H:%M:%S.%f")
            deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                           microseconds=time.microsecond)
            return deltatime

    # 用于打包数据成列表
    def zip1(self):
        data = [self.entry1.get(), self.entry2.get(), self.entry3.get(), self.entry4.get()]
        for i in data:
            if len(i) == 0:
                showwarning(title="WARNING", message='Cant be null')
        #print(type(data))
        self.text.insert('end', data)
        self.text.insert('end', '\n')

    # 打包好格式
    def form_data(self):
        #print(self.process)
        self.process.to_csv(path_or_buf='./Result.csv',index = False)
        self.training_data.to_csv(path_or_buf='./training_data.csv')






root = tk.Tk()
root.title('hens')
root.geometry('1281x640')

apps = app(master=root)
root.mainloop()
