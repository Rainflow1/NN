import tkinter.font
from tkinter import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from prepare_data import process_dataset
from argparse import *
from Trainer import Trainer
from PIL import Image, ImageTk
import sys
import threading


class PrintLogger(object):  # create file like object

    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly

    def flush(self):  # needed for file like object
        pass


class Window:

    def redirect_logging(self, wigited):
        logger = PrintLogger(wigited)
        sys.stdout = logger
        sys.stderr = logger

    def reset_logging(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def __init__(self):
        root = Tk()
        self.master = root
        self.master.title("Zliczanie ludzi w tłumie")
        self.master.geometry("1000x560")
        self.master.columnconfigure(0, weight=1, uniform="col")
        self.master.rowconfigure(1, weight=1)
        self.frame = Frame(self.master)
        self.font = tkinter.font.nametofont("TkTextFont")

        self.frameName = ""
        self.datasetPath = ""
        self.datasetResultPath = ""
        self.trainDatasetPath = ""
        self.trainCheckpointPath = ""
        self.trainLr = 0.00001
        self.trainWeighDecay = 0.0001
        self.trainEpochs = 500
        self.trainCropSize = 512
        self.testImagePath = ""
        self.testModelPath = ""

        taskbar = Frame(self.master, bg="white")
        taskbar.grid(row=0, column=0, columnspan=2, sticky='wens')
        taskbar.columnconfigure(0, weight=1, uniform="bar_col")
        taskbar.columnconfigure(1, weight=1, uniform="bar_col")
        taskbar.columnconfigure(2, weight=1, uniform="bar_col")
        taskbar.columnconfigure(3, weight=6, uniform="bar_col")

        nazwy = ["Baza zdjęć", "Trenowanie", "Oblicz liczbę osób", ""]
        for i in range(4):
            label = Button(taskbar, text=nazwy[i], height=2, width=15, highlightbackground="white",
                           state=("disabled" if i == 3 else "active"))
            label.grid(row=0, column=i, sticky="wens")
            if i == 0:
                label.configure(command=lambda: self.clearAndCreateFrame("Settings", 0))
            if i == 1:
                label.configure(command=lambda: self.clearAndCreateFrame("Train", 1))
            if i == 2:
                label.configure(command=lambda: self.clearAndCreateFrame("Test", 2))

        print(self.frameName)
        root.mainloop()

    def setVariable(self, var, val):
        if var == "datasetPath":
            self.datasetPath = val
        if var == "datasetResultPath":
            self.datasetResultPath = val
        if var == "trainDatasetPath":
            self.trainDatasetPath = val
        if var == "trainCheckpointPath":
            self.trainCheckpointPath = val
        if var == "testImagePath":
            self.testImagePath = val
        if var == "testModelPath":
            self.testModelPath = val

    def browseModel(self, var):
        filename = filedialog.askopenfilename(initialdir="/", title="Wybierz model", filetypes=[("Model", "*.*")])
        self.setVariable(var, filename)

    def browseImages(self, var):
        filename = filedialog.askopenfilename(initialdir="/", title="Wybierz obraz",
                                              filetypes=[("Image", "*.png"), ("Image", "*.jpg")])
        self.setVariable(var, filename)

    def browseDirs(self, var):
        dirname = filedialog.askdirectory(initialdir="/", title="Wybierz folder")
        self.setVariable(var, dirname)

    def setText(self, obj, var):
        if var == "":
            obj.delete("1.0", END)
            obj.insert(INSERT, "")
        else:
            obj.delete("1.0", END)
            obj.insert(INSERT, var)

    def clearAndCreateFrame(self, name, func):
        #print(self.frameName + " " + name)
        if self.frameName != name:
            self.frame.grid_forget()
            self.reset_logging()
            if func == 0:
                self.createSettingsFrame()
            if func == 1:
                self.createTrainFrame()
            if func == 2:
                self.createTestFrame()

    def threadDatasetProcess(self):
        thread = threading.Thread(target=process_dataset, args=(self.datasetPath, self.datasetResultPath))
        thread.start()

    def createSettingsFrame(self):
        frame = Frame(self.master)
        frame.grid(row=1, column=0, sticky="wens")
        self.frame = frame
        self.frameName = "Settings"

        Label(frame, text="Ścieżka do zbioru danych").place(x=50, y=30)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseDirs("datasetPath"), self.setText(dataset_path, self.datasetPath),
                                      ], padx=10)
        btn.place(x=338, y=22)

        dataset_path = Text(frame, width=65, height=2, wrap=WORD, font=self.font)
        dataset_path.insert(INSERT, self.datasetPath)
        dataset_path.place(x=50, y=70)

        Label(frame, text="Ścieżka do wynikowego folderu").place(x=50, y=150)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseDirs("datasetResultPath"), self.setText(dataset_result_path, self.datasetResultPath),
                                      ], padx=10)
        btn.place(x=338, y=142)

        dataset_result_path = Text(frame, width=65, height=2, font=self.font)
        dataset_result_path.insert(INSERT, self.datasetResultPath)
        dataset_result_path.place(x=50, y=190)

        btn = Button(frame, text="Przetwórz bazę zdjęć", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.redirect_logging(scrolled_text),
                                      self.threadDatasetProcess()], padx=10)
        btn.place(x=167, y=252)

        scrolled_text = ScrolledText(frame, height=25, width=80, wrap=CHAR, font=self.font)
        scrolled_text.place(x=475, y=30)

    def threadTraining(self):
        thread = threading.Thread(target=self.trainingInThread)
        thread.start()

    def trainingInThread(self):
        parser = ArgumentParser(description="Training")
        parser.add_argument("--dataset", default=str(self.trainDatasetPath), help="preprocessed dataset path")
        parser.add_argument("--model", default=str(self.trainCheckpointPath), help="model save path")
        parser.add_argument("--lr", type=float, default=float(self.learnRateEntry.get()), help="learn rate")
        parser.add_argument("--weight-decay", type=float, default=float(self.weightDecayEntry.get()),
                            help="weight decay")
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--epochs", type=int, default=int(self.epochsEntry.get()))
        parser.add_argument("--crop-size", type=int, default=int(self.cropSizeEntry.get()))
        args = parser.parse_args()
        trainer = Trainer(args)
        trainer.train()
    
    def createTrainFrame(self):
        frame = Frame(self.master)
        frame.grid(row=1, column=0, sticky="wens")
        self.frame = frame
        self.frameName = "Train"

        Label(frame, text="Scieżka do przetworzonego zbioru danych").place(x=50, y=30)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseDirs("trainDatasetPath"), self.setText(train_dataset_path, self.trainDatasetPath),
                                      ], padx=10)
        btn.place(x=338, y=22)

        train_dataset_path = Text(frame, width=65, height=2, wrap=WORD, font=self.font)
        train_dataset_path.insert(INSERT, self.trainDatasetPath)
        train_dataset_path.place(x=50, y=70)

        Label(frame, text="Scieżka do zapisania modelu").place(x=50, y=150)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseDirs("trainCheckpointPath"), self.setText(train_checkpoint_path, self.trainCheckpointPath),
                                      ], padx=10)
        btn.place(x=338, y=142)

        train_checkpoint_path = Text(frame, width=65, height=2, wrap=WORD, font=self.font)
        train_checkpoint_path.insert(INSERT, self.trainDatasetPath)
        train_checkpoint_path.place(x=50, y=190)

        Label(frame, text="Learning rate").place(x=50, y=250)
        self.learnRateEntry = Entry(frame, width=40)
        self.learnRateEntry.place(x=200, y=250)

        Label(frame, text="Weight decay").place(x=50, y=280)
        self.weightDecayEntry = Entry(frame, width=40)
        self.weightDecayEntry.place(x=200, y=280)

        Label(frame, text="Epochs").place(x=50, y=310)
        self.epochsEntry = Entry(frame, width=40)
        self.epochsEntry.place(x=200, y=310)

        Label(frame, text="Crop size").place(x=50, y=340)
        self.cropSizeEntry = Entry(frame, width=40)
        self.cropSizeEntry.place(x=200, y=340)

        btn = Button(frame, text="Rozpocznij trenowanie", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.redirect_logging(train_scrolled_text), self.threadTraining()], padx=10)
        btn.place(x=180, y=390)

        train_scrolled_text = ScrolledText(frame, height=25, width=80, wrap=CHAR, font=self.font)
        train_scrolled_text.place(x=475, y=30)

    def updateImg(self):
        path = self.testImagePath
        if path == "":
            self.img.place_forget()
        else:
            image = Image.open(path)
            resized_img = image.resize((500, 280))
            img = ImageTk.PhotoImage(resized_img)
            self.img.configure(image=img)
            self.img.image = img
            self.img.place(x=5, y=5)

    def threadCountOneImage(self):
        thread = threading.Thread(target=self.countInThread)
        thread.start()

    def countInThread(self):
        modelPath = str(self.testModelPath)
        imagePath = str(self.testImagePath)
        Trainer.testOneImage(modelPath, imagePath)


    def createTestFrame(self):
        frame = Frame(self.master)
        frame.grid(row=1, column=0, sticky="wens")
        self.frame = frame
        self.frameName = "Test"

        Label(frame, text="Scieżka do zdjęcia").place(x=50, y=30)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseImages("testImagePath"),
                                      self.setText(test_img_path, self.testImagePath),
                                      self.updateImg()], padx=10)
        btn.place(x=338, y=22)

        test_img_path = Text(frame, width=65, height=2, wrap=WORD, font=self.font)
        test_img_path.insert(INSERT, self.testImagePath)
        test_img_path.place(x=50, y=70)

        Label(frame, text="Scieżka do wytrenowanego modelu").place(x=50, y=150)
        btn = Button(frame, text="Przeglądaj pliki", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.browseModel("testModelPath"),
                                      self.setText(test_model_path, self.testModelPath)], padx=10)
        btn.place(x=338, y=142)

        test_model_path = Text(frame, width=65, height=2, wrap=WORD, font=self.font)
        test_model_path.insert(INSERT, self.testImagePath)
        test_model_path.place(x=50, y=190)

        img_bg = Frame(frame, width=515, height=295, bg="#c5d8e8")
        img_bg.place(x=460, y=100)
        self.img = Label(img_bg)

        btn = Button(frame, text="Oblicz ilośc osób", height=2, borderwidth=1, relief="solid",
                     command=lambda: [self.redirect_logging(test_scrolled_text), self.threadCountOneImage()], padx=10)
        btn.place(x=190, y=240)

        test_scrolled_text = ScrolledText(frame, height=10, width=30, wrap=CHAR, font=self.font)
        test_scrolled_text.place(x=150, y=290)



if __name__ == "__main__":
    window = Window()

