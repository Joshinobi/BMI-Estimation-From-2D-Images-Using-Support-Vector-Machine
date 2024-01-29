from fnmatch import fnmatchcase
from xml.sax.handler import feature_namespace_prefixes
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import sys
import os

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        #Load the ui file
        uic.loadUi("/home/pi/Desktop/thesis_program/thesis_main.ui", self)

        #Define our widgets
        self.button1 = self.findChild(QPushButton, "pushButton1")
        self.button2 = self.findChild(QPushButton, "pushButton2")
        self.button3 = self.findChild(QPushButton, "pushButton3")
        self.label1 = self.findChild(QLabel, "imageDisplay")
        self.label2 = self.findChild(QLabel, "estimateBMI")

        #Click functions
        self.button1.clicked.connect(self.takePhoto)
        self.button2.clicked.connect(self.selectImage)
        self.button3.clicked.connect(self.estimate)

        #Show the app
        self.show()
    
    def takePhoto(self):
        os.system("raspistill -o /home/pi/Desktop/thesis_program/image01.jpg")

    def selectImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "/home/pi/Desktop/thesis_program")
        #Open the image
        self.pixmap = QPixmap(fname[0])
        #Add pic to label
        self.label1.setPixmap(self.pixmap)

        import numpy as np
        import cv2
        from scipy.spatial.distance import euclidean
        import mtcnn

        #Face extraction and alignment
        eye_pos_w, eye_pos_h = 0.35, 0.4
        width, height = 500, 500

        img = cv2.imread(fname[0])
        h, w, _ = img.shape

        img_rgb = img[..., ::-1]
        face_detector = mtcnn.MTCNN()
        results = face_detector.detect_faces(img_rgb)

        l_e = results[0]['keypoints']['left_eye']
        r_e = results[0]['keypoints']['right_eye']
        center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

        dx = (r_e[0] - l_e[0])
        dy = (r_e[1] - l_e[1])
        dist = euclidean(l_e, r_e)

        angle = np.degrees(np.arctan2(dy, dx)) + 360
        scale = width * (1 - (2 * eye_pos_w)) / dist

        tx = width * 0.5
        ty = height * eye_pos_h

        m = cv2.getRotationMatrix2D(center, angle, scale)

        m[0, 2] += (tx - center[0])
        m[1, 2] += (ty - center[1])

        global face_align
        face_align = cv2.warpAffine(img, m, (width, height))

        cv2.waitKey(0)

    def estimate(self):

        import numpy as np
        import cv2
        import mediapipe as mp
        import math
        import pandas as pd
        import sklearn
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, plot_confusion_matrix
        from sklearn.utils import resample
        import warnings
        warnings.filterwarnings("ignore")

        status = cv2.imwrite('/home/pi/Desktop/thesis_program/image01_extracted.jpg', face_align)

        #Facial feature extraction
        mp_face_mesh=mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        image=cv2.imread('/home/pi/Desktop/thesis_program/image01_extracted.jpg')
        rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(image)
        height, width, _ = image.shape

        for facial_landmarks in result.multi_face_landmarks:

                f = open('/home/pi/Desktop/thesis_program/image01_ratio.csv', 'a')

                #CJWR
                p1 = facial_landmarks.landmark[127] 
                xp1 = p1.x
                yp1 = p1.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p1p15_1 = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                p4p12_1 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                cjwr = str(p1p15_1 / p4p12_1)

                #WHR
                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p67 = facial_landmarks.landmark[13]
                xp67 = p67.x
                yp67 = p67.y

                n1 = facial_landmarks.landmark[8]
                xn1 = n1.x
                yn1 = n1.y

                p4p12_2 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                p67n1 = math.sqrt((xn1 - xp67)**2 + (yn1 - yp67)**2)
                whr = str(p4p12_2 / p67n1)
                
                #PAR
                p1 = facial_landmarks.landmark[127]
                xp1 = p1.x
                yp1 = p1.y

                p4 = facial_landmarks.landmark[58]
                xp4 = p4.x
                yp4 = p4.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                p12 = facial_landmarks.landmark[367]
                xp12 = p12.x
                yp12 = p12.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                p67 = facial_landmarks.landmark[13]
                xp67 = p67.x
                yp67 = p67.y

                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                per1 = math.sqrt((xp4 - xp1)**2 + (yp4 - yp1)**2)
                per2 = math.sqrt((xp8 - xp4)**2 + (yp8 - yp4)**2)
                per3 = math.sqrt((xp12 - xp8)**2 + (yp12 - yp8)**2)
                per4 = math.sqrt((xp15 - xp12)**2 + (yp15 - yp12)**2)
                per5 = math.sqrt((xp1 - xp15)**2 + (yp1 - yp15)**2)
                perimeter = per1 + per2 + per3 + per4 + per5

                a = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                b_1 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                h_1 = math.sqrt((xp67 - xn5)**2 + (yp67 - yn5)**2)
                a1 = ((a + b_1) / 2) * h_1

                h_2 = math.sqrt((xp8 - xp67)**2 + (yp8 - yp67)**2)
                b_2 = math.sqrt((xp12 - xp4)**2 + (yp12 - yp4)**2)
                a2 =   (h_2 * b_2) / 2

                area = a1 + a2

                par = str(perimeter / area)

                #ES
                p28 = facial_landmarks.landmark[33]
                xp28 = p28.x
                yp28 = p28.y

                p33 = facial_landmarks.landmark[359]
                xp33 = p33.x
                yp33 = p33.y

                p30 = facial_landmarks.landmark[112]
                xp30 = p30.x
                yp30 = p30.y

                p35 = facial_landmarks.landmark[362]
                xp35 = p35.x
                yp35 = p35.y

                p28p33 = math.sqrt((xp33 - xp28)**2 + (yp33 - yp28)**2)
                p30p35 = math.sqrt((xp35 - xp30)**2 + (yp35 - yp30)**2)
                es = str((p28p33 - p30p35) / 2)

                #LF/HR
                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                n2 = facial_landmarks.landmark[10]
                xn2 = n2.x
                yn2 = n2.y

                lfh = math.sqrt((xp8 - xn5)**2 + (yp8 - yn5)**2)
                n2p8 = math.sqrt((xp8 - xn2)**2 + (yp8 - yn2)**2)
                lfhr = str(lfh / n2p8)

                #FW/LFH
                p1 = facial_landmarks.landmark[127]
                xp1 = p1.x
                yp1 = p1.y

                p15 = facial_landmarks.landmark[264]
                xp15 = p15.x
                yp15 = p15.y

                n5 = facial_landmarks.landmark[168]
                xn5 = n5.x
                yn5 = n5.y

                p8 = facial_landmarks.landmark[152]
                xp8 = p8.x
                yp8 = p8.y

                p1p15_2 = math.sqrt((xp15 - xp1)**2 + (yp15 - yp1)**2)
                lfh = math.sqrt((xp8 - xn5)**2 + (yp8 - yn5)**2)
                fwlfh = str(p1p15_2 / lfh)

                #MEH
                p22 = facial_landmarks.landmark[70]
                xp22 = p22.x
                yp22 = p22.y

                p28 = facial_landmarks.landmark[33]
                xp28 = p28.x
                yp28 = p28.y

                n3 = facial_landmarks.landmark[52]
                xn3 = n3.x
                yn3 = n3.y

                p29 = facial_landmarks.landmark[27]
                xp29 = p29.x
                yp29 = p29.y

                p25 = facial_landmarks.landmark[221]
                xp25 = p25.x
                yp25 = p25.y

                p30 = facial_landmarks.landmark[112]
                xp30 = p30.x
                yp30 = p30.y

                p19 = facial_landmarks.landmark[285]
                xp19 = p19.x
                yp19 = p19.y

                p35 = facial_landmarks.landmark[362]
                xp35 = p35.x
                yp35 = p35.y

                n4 = facial_landmarks.landmark[334]
                xn4 = n4.x
                yn4 = n4.y

                p34 = facial_landmarks.landmark[386]
                xp34 = p34.x
                yp34 = p34.y

                p16 = facial_landmarks.landmark[300]
                xp16 = p16.x
                yp16 = p16.y

                p33 = facial_landmarks.landmark[359]
                xp33 = p33.x
                yp33 = p33.y

                p22p28 = math.sqrt((xp28 - xp22)**2 + (yp28 - yp22)**2)
                n3p29 = math.sqrt((xp29 - xn3)**2 + (yp29 - yn3)**2)
                p25p30 = math.sqrt((xp30 - xp25)**2 + (yp30 - yp25)**2)
                p19p35 = math.sqrt((xp35 - xp19)**2 + (yp35 - yp19)**2)
                n4p34 = math.sqrt((xp34 - xn4)**2 + (yp34 - yn4)**2)
                p16p33 = math.sqrt((xp33 - xp16)**2 + (yp33 - yp16)**2)

                meh = str((p22p28 + n3p29 + p25p30 + p19p35 + n4p34 + p16p33) / 6)

                f.write(cjwr + ', ')
                f.write(whr + ', ')
                f.write(par + ', ')
                f.write(es + ', ')
                f.write(lfhr + ', ')
                f.write(fwlfh + ', ')
                f.write(meh + '\n')

                f.close()

        cv2.waitKey(0)

        def load_model():
            try:
                with open("/home/pi/Desktop/thesis_program/bestModel.pickle", "rb") as f:
                    model, accuracy, X_test, y_test = pickle.load(f)
                    f.close
                load_model.best = accuracy
                return model, accuracy, X_test, y_test
            except OSError:
                load_model.best = 0
                print("No trained model found")
                pass


        load_model()
        best = load_model.best

        def load_model2(name):
            try:
                with open(name, "rb") as f:
                    model, accuracy, X_test, y_test = pickle.load(f)
                    f.close
                load_model.best = accuracy
                return model, accuracy, X_test, y_test
            except OSError:
                load_model.best = 0
                print("No trained model found")
                pass

        def resampling(df, s):
            normal_df = resample(df.loc[df['BMI']==0], replace=True, n_samples=s)
            obese_df = resample(df.loc[df['BMI']==1], replace=True, n_samples=s)
            overweight_df = resample(df.loc[df['BMI']==2], replace=True, n_samples=s)
            underweight_df = resample(df.loc[df['BMI']==3], replace=True, n_samples=s)
            df = pd.DataFrame(columns=['CJWR', 'WHR', 'PAR', 'ES', 'LF/FH', 'FW/LFH', 'MEH', 'BMI'])
            df = underweight_df.append(overweight_df.append(obese_df.append(normal_df, ignore_index=True), ignore_index=True), ignore_index=True)
            df = df.dropna(how='any', inplace=False)
            df = df.sample(frac=1).reset_index(drop=True)
            return df

        def data_split(df):
            X = df.loc[:, df.columns != 'BMI']
            y = df.loc[:, df.columns == 'BMI']
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            return X_train, X_test, y_train, y_test

        def scale_data(X_train, X_test):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.astype(np.float32))
            X_test = scaler.transform(X_test.astype(np.float32))
            return X_train, X_test


        def predict(pred, fpred):
            if pred == 0:
                fpred[0] += 1
            elif pred == 1:
                fpred[1] += 1
            elif pred == 2:
                fpred[2] += 1
            elif pred == 3:
                fpred[3] += 1
            print(fpred)
            return fpred

        def translate(pred, fpred, trans):
            if pred == 0:
                fpred = predict(trans, fpred)
            else:
                for i in range(4):
                    if i == trans:
                        pass
                    else:
                        fpred = predict(i, fpred)
            return fpred

        normal_df = pd.read_csv("/home/pi/Desktop/thesis_program/training_dataset/normal_dataset.csv")
        normal_df = normal_df.dropna(how='any', inplace=False)
        obese_df = pd.read_csv("/home/pi/Desktop/thesis_program/training_dataset/obese_dataset.csv")
        obese_df = obese_df.dropna(how='any', inplace=False)
        overweight_df = pd.read_csv("/home/pi/Desktop/thesis_program/training_dataset/overweight_dataset.csv")
        overweight_df = overweight_df.dropna(how='any', inplace=False)
        underweight_df = pd.read_csv("/home/pi/Desktop/thesis_program/training_dataset/underweight_dataset.csv")
        underweight_df = underweight_df.dropna(how='any', inplace=False)
        df_orig = underweight_df.append(overweight_df.append(obese_df.append(normal_df, ignore_index=True), ignore_index=True), ignore_index=True)
        df = df_orig.dropna(how='any', inplace=False)
        model, accuracy, X_test, y_test = load_model()
        y_pred = model.predict(X_test)

        classification_report = classification_report(y_test, y_pred)
        accuracy = model.score(X_test, y_test)

        disp = plot_confusion_matrix(model, X_test, y_test, normalize='true')
        disp.plot

        x1, x2, y1, y2 = data_split(df)
        x1 = x1.reset_index(drop=True)
        fpred = [0, 0, 0, 0]

        file = open('/home/pi/Desktop/thesis_program/image01_ratio.csv', 'r')
        lines = file.readlines()
        test = lines[0].split(",")
        CJWR, WHR, PAR, ES, LFFH, FWLFH, MEH = test
        inputs2 = [CJWR, WHR, PAR, ES, LFFH, FWLFH, MEH]
        inputs = [float(i) for i in inputs2]
        inputs_df = pd.DataFrame(columns=['CJWR', 'WHR', 'PAR', 'ES', 'LF/FH', 'FW/LFH', 'MEH'])
        inputs_df.loc[0] = inputs
        inputs_df
        scaler = StandardScaler()
        scaler.fit_transform(x1.astype(np.float32))
        scaled_inputs = scaler.transform(inputs_df.astype(np.float32))

        voting = 10
        for i in range(voting):
            model, accuracy, X_test, y_test = load_model2("/home/pi/Desktop/thesis_program/bestModel2_1.pickle")
            pred = model.predict(scaled_inputs)
            fpred = translate(pred, fpred, 0)
            model, accuracy, X_test, y_test = load_model2("/home/pi/Desktop/thesis_program/bestModel2_2.pickle")
            pred = model.predict(scaled_inputs)
            fpred = translate(pred, fpred, 1)
            model, accuracy, X_test, y_test = load_model2("/home/pi/Desktop/thesis_program/bestModel2_3.pickle")
            pred = model.predict(scaled_inputs)
            fpred = translate(pred, fpred, 2)
            model, accuracy, X_test, y_test = load_model2("/home/pi/Desktop/thesis_program/bestModel2_4.pickle")
            pred = model.predict(scaled_inputs)
            fpred = translate(pred, fpred, 3)
            model, accuracy, X_test, y_test = load_model()
            pred = model.predict(scaled_inputs)
            fpred = predict(pred, fpred)

        model, accuracy, X_test, y_test = load_model()
        pred = model.predict(scaled_inputs)
        fpred = predict(pred, fpred)

        high = 0
        for i in range(4):
            if fpred[i] > high:
                index = i
                high = fpred[i]
            else:
                pass
        fpred = index

        #os.remove("/home/gonzales/Thesis/main_program02/test01_extracted.jpg")
        #os.remove("/home/gonzales/Thesis/main_program02/test01_ratio.csv")

        if fpred == 0:
            self.label2.setText("Normal")
        elif fpred == 1:
            self.label2.setText("Obese")
        elif fpred == 2:
            self.label2.setText("Overweight")
        elif fpred == 3:
            self.label2.setText("Underweight")

#Initialize the app
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
