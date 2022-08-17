#TEXT DETECTION Using Neural Network

from re import I
from tkinter import *
from PIL import ImageTk,Image, UnidentifiedImageError
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras_ocr
import re


root = Tk()
root.title('Text Conversion')
root.geometry("720x720")

pipeline = keras_ocr.pipeline.Pipeline()


def openfile():
    global filenamelabel,img
    root.filename = filedialog.askopenfilename(initialdir="D:",title="open file")
    if root.filename == "":
        filenamelabel = "Error"
        convertbutton["state"] = "disabled"
        detect_button["state"] = "disabled"
    else:
        filenamelabel = root.filename
        convertbutton["state"] = "normal"
        detect_button["state"] = "normal"
    filelabel["text"] = filenamelabel


    try:
        img = Image.open(filenamelabel)
        re_img = img.resize((480,480),Image.ANTIALIAS)        
        tk_img = ImageTk.PhotoImage(re_img)
    except (UnidentifiedImageError,FileNotFoundError):
        img = Image.open("D:/New folder/Robo_Prgm/project/default.jfif")
        re_img = img.resize((480,480),Image.ANTIALIAS)
        tk_img = ImageTk.PhotoImage(re_img)
        convertbutton["state"] = "disabled"
        detect_button["state"] = "disabled"
    viewer["image"] = tk_img
    viewer.image = tk_img

#detection
def detect():
    images = [keras_ocr.tools.read(filenamelabel)]
    
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    axs = [axs]
    
    # Plot the predictions
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    plt.savefig('saved_figure.png')
    img = Image.open("saved_figure.png")
    re_img = img.resize((480,480),Image.ANTIALIAS)
    tk_img = ImageTk.PhotoImage(re_img)
    viewer["image"] = tk_img
    viewer.image = tk_img
    global textDetected
    textDetected = ""
    for i in prediction_groups[0]:
        textDetected = textDetected+" "+ i[0]

#preprocessing 
def preprocesing():
    img_cv = cv2.imread(filenamelabel)
    
    gray_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    #creating Binary image by selecting proper threshold
    binary_image = cv2.threshold(gray_image ,130,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
 
    #Inverting the image
    inverted_bin = cv2.bitwise_not(binary_image)
 
    #Some noise reduction
    kernel = np.ones((2,2),np.uint8)
    processed_img = cv2.erode(inverted_bin, kernel, iterations = 1)
    processed_img = cv2.dilate(processed_img, kernel, iterations = 1)

    #cv2.imshow('',processed_img)

    return processed_img
def clean(text):
        return re.sub('[^A-Za-z0-9" "]+', ' ', text)
def convert():
    # Data cleaning
    text_otp = clean(textDetected)
    result_text.set(textDetected)

#Default image
defaultimg = Image.open("D:/New folder/Robo_Prgm/project/default.jfif")
defaultimg = defaultimg.resize((480,480),Image.ANTIALIAS)
defaultimg = ImageTk.PhotoImage(defaultimg)

#Creating Labels & Buttons
result_text = StringVar()
result_text.set("output")

frame0 = LabelFrame(root,padx=10,pady=10)
frame1 = LabelFrame(root,padx=10,pady=10) #Creating Frame for viewing image
viewer = Label(frame1,image=defaultimg)
viewer.image = defaultimg

openButton = Button(frame0,text="Open file",command=openfile)
convertbutton = Button(frame0,text="Convert",command=convert,state="disabled")
detect_button = Button(frame0,text="Detect",command=detect,state="disabled")

result = Entry(root,textvariable=result_text)
filelabel = Label(root,text="Open File")

#Packing
frame0.place(x=10,y=1,width=700)
openButton.grid(row=0,column=0)
filelabel.place(x=10,y=50,width=700)
frame1.place(x=10,y=100,width=700)
viewer.pack()
detect_button.grid(row=0,column=1)
convertbutton.grid(row=0,column=2)
result.place(x=10,y = 600,width=700,height=100)

root.mainloop()