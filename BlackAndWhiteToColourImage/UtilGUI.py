from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import black_and_white_to_color as bawc

interface = Tk()
interface.geometry("400x300")
inputVideoPath = None
outputVideoPath = ''
outputVideoNameText = StringVar()
outputVideoNameText.set('MyColoredRender')

def openfile():
    global inputVideoPath
    inputVideoPath = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes = (("Video files", "*.mp4*"), ("all files", "*.*")))
    return inputVideoPath

def browse_button():
    global outputVideoPath
    outputVideoPath = filedialog.askdirectory()

def initiate_convert():
    global outputVideoNameText
    output_path = outputVideoPath + '/' + outputVideoNameText.get() + ".mp4"
    #print(output_path)
    if inputVideoPath == None:
        print("No video selected")
        return
    else:
        print('Initiate Conversion!')
        bawc.colorBAWVideo(inputVideoPath, output_path)
        #progress_text = Label(interface, text = ".").place(x = 150, y = 200) 
        #progress_text["text"] = str(bawc.get_progress)

user_name = Label(interface, text = "Colorize Black and White Video").place(x = 10, y = 20) 

inputVideo = Button(interface, text="Select Input Video", command=openfile).place(x = 60, y = 60)
outputVideo = Button(interface, text="Select Output Folder", command=browse_button).place(x = 220, y = 60)


video_name_label = Label(interface, text = "Output Video Name").place(x = 20, y = 110)
output_video_name = Entry(interface, textvariable = outputVideoNameText, width = 30).place(x = 160, y = 110)

submit_button = Button(interface, text="Convert", command=initiate_convert).place(x = 150, y = 160)

interface.mainloop()