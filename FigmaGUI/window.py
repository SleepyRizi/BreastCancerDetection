from tkinter import *
import tkinter
from PIL import ImageTk,Image #pillow
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import cv2 #preprocessing
import numpy as np #computation
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.linear_model import LogisticRegression
import pickle


global imgTest,medianImg,roiImage,formatted,edged_img,box_name,robertImg_final



#################### Functions ############################

def upload_file():
    global imgTest,testImg,imgresized
    f_types = [('PGM Files', '*.pgm')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    imgTest=Image.open(filename)

    imgresized = imgTest.resize((151,152))
    imgresized= ImageTk.PhotoImage(imgresized)
    testImg_placed=canvas.create_image(246,161,image=imgresized)



def preprocessing(imgTest):
    global medianImg,roiImage,robertsImg,formatted,edged_img
    width,height=550,731
    x,y=250,276
    # print(imgTest)
    imgTest= np.uint8(imgTest)
    medianImg= cv2.medianBlur(imgTest,11)
    croped=medianImg[y:y+height, x:x+width]

    #---------------------------------------#
    roberts_cross_v = np.array( [[1, 0 ],
                                [0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 1 ],
                                [ -1, 0 ]] )
    vertical = ndimage.convolve(croped, roberts_cross_v )
    horizontal = ndimage.convolve(croped, roberts_cross_h )
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    formatted = (edged_img * 255 / np.max(edged_img)).astype('uint8')

    edged_img=roberts_filter(croped)

    #---------------------------------------#

    medianImg = cv2.resize(medianImg, dsize=(151, 152), interpolation=cv2.INTER_CUBIC)
    croped = cv2.resize(croped, dsize=(151, 152), interpolation=cv2.INTER_CUBIC)
    roberts= cv2.resize(formatted,dsize=(151,152),interpolation=cv2.INTER_CUBIC)

    medianImg = Image.fromarray(np.uint8(medianImg))
    croped= Image.fromarray(np.uint8(croped))
    roberts= Image.fromarray(roberts)


    medianImg= ImageTk.PhotoImage(medianImg)
    roiImage= ImageTk.PhotoImage(croped)
    robertsImg= ImageTk.PhotoImage(roberts)

    medianImg_placed=canvas.create_image(406,161,image=medianImg)
    roi_placed=canvas.create_image(566,161,image=roiImage)
    robertsImg_placed=canvas.create_image(726,161,image=robertsImg)

def roberts_filter(img):
    global robertImg_final
    roberts_cross_v = np.array( [[1, 0 ],
                                [0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 1 ],
                                [ -1, 0 ]] )
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    robertImg_final = np.sqrt( np.square(horizontal) + np.square(vertical))
    robertImg_final=robertImg_final.astype(np.uint8)





def feature_extraction(robertImg_final):
    no =[]
    global temp_dic
    temp_dic={}
    features=['contrast','correlation','energy','homogeneity']
    distances=[1,3,5]
    angles=[0,np.pi/4,np.pi/2]
    GLCM = graycomatrix(robertImg_final,distances,angles)

    for n in features:
        for j in range(0,len(angles)):
            for k in range(0,len(distances)):
                col_values=graycoprops(GLCM,n)[j][k]
                col_name=n[:3]+'_'+str(int(np.degrees(angles[j])))+'_'+str(distances[k])
                temp_dic.update({col_name:col_values})

    #print(temp_dic)
    for i in range(0,36):
        box_name= 'entry'+str(i)
        #entry.
        globals()[box_name].delete(0, 'end')
        no.append(i)

    for keys,j in zip(temp_dic,no):

        box_name= 'entry'+str(j)
        #globals()[box_name].clear()
        #print(box_name,temp_dic[keys])

        globals()[box_name].insert('end',temp_dic[keys])





    # #print(temp_dic)

def predict_class(temp_dic):


    classifier= pickle.load(open("cancerdetection.pkl",'rb'))
    classifierMandB= pickle.load(open("M_nd_B.pkl",'rb'))
    single_img=np.reshape(list(temp_dic.values()),(-1,36))
    prediction=classifier.predict(single_img)[0]

    if prediction==0.0:
        cancer='Normal'
    else:
        prediction=classifierMandB.predict(single_img)[0]
        if prediction==0.0:
            cancer='Benign'
        else:
            cancer='Malignant'



    globals()['entry36'].delete(0, 'end')
    globals()['entry37'].delete(0, 'end')

    globals()['entry36'].insert('end',prediction)

    globals()['entry37'].insert('end',cancer)




###########################################################


window = Tk()
window.geometry("1012x770")
window.title("Automated Breast Cancer Detection")
window.configure(bg = "#f9fbe3")
logo = PhotoImage(file = 'icon.png')
window.iconphoto(False, logo)
canvas = Canvas(
    window,
    bg = "#f9fbe3",
    height = 770, #770
    width = 1012, #1012
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    505, 385,
    image=background_img)

############## Images ##############################33
testImg= Image.open('whiteImage.png')
testImg=testImg.resize((151,152))
testImg= ImageTk.PhotoImage(testImg)
testImg_placed=canvas.create_image(246,161,image=testImg)

medianImg= Image.open('whiteImage.png')
medianImg=medianImg.resize((151,152))
medianImg= ImageTk.PhotoImage(medianImg)
medianImg_placed=canvas.create_image(406,161,image=medianImg)


roiImage= Image.open('whiteImage.png')
roiImage=roiImage.resize((151,152))
roiImage= ImageTk.PhotoImage(roiImage)
roi_placed=canvas.create_image(566,161,image=roiImage)


robertsImg= Image.open('whiteImage.png')
robertsImg=robertsImg.resize((151,152))
robertsImg= ImageTk.PhotoImage(robertsImg)
robertsImg_placed=canvas.create_image(726,161,image=robertsImg)

#################### Buttons #######################


browse_img = PhotoImage(file = f"img2.png")
browseBtn = Button(
    image = browse_img,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda:upload_file(),
    relief = "flat")

browseBtn.place(
    x = 14, y = 89,
    width = 142,
    height = 32)
# ------------------------------------------------


preprocess_img = PhotoImage(file = f"img3.png")
preprocessBtn = Button(
    image = preprocess_img,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda:preprocessing(imgTest),
    relief = "flat")

preprocessBtn.place(
    x = 14, y = 127,
    width = 142,
    height = 32)
# ------------------------------------------------

feature_img = PhotoImage(file = f"img1.png")
featureBtn = Button(
    image = feature_img,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda:feature_extraction(robertImg_final),
    relief = "flat")

featureBtn.place(
    x = 14, y = 167,
    width = 142,
    height = 33)

# ------------------------------------------------

predict_img = PhotoImage(file = f"img0.png")
predictBtn = Button(
    image = predict_img,
    borderwidth = 0,
    highlightthickness = 0,
    command = lambda: predict_class(temp_dic),
    relief = "flat")

predictBtn.place(
    x = 14, y = 205,
    width = 142,
    height = 32)

##################################################

################## Entry Box #####################

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    1000, 0,
    image = entry0_img)

entry0 = Entry(

    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)
entry0.place(
    x = 329, y = 315,
    width = 108,
    height = 23)

entry1_img = PhotoImage(file = f"img_textBox1.png")
entry1_bg = canvas.create_image(
    0,0,
    image = entry1_img)

entry1 = Entry(


    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry1.place(
    x = 329, y = 353,
    width = 108,
    height = 23)

entry2_img = PhotoImage(file = f"img_textBox2.png")
entry2_bg = canvas.create_image(
    0,0,
    image = entry2_img)

entry2 = Entry(

    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry2.place(
    x = 329, y = 391,
    width = 108,
    height = 23)

entry3_img = PhotoImage(file = f"img_textBox3.png")
entry3_bg = canvas.create_image(
    0,0,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry3.place(
    x = 329, y = 432,
    width = 108,
    height = 23)

entry4_img = PhotoImage(file = f"img_textBox4.png")
entry4_bg = canvas.create_image(
    0,0,
    image = entry4_img)

entry4 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry4.place(
    x = 329, y = 466,
    width = 108,
    height = 23)

entry5_img = PhotoImage(file = f"img_textBox5.png")
entry5_bg = canvas.create_image(
    0,0,
    image = entry5_img)

entry5 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry5.place(
    x = 329, y = 505,
    width = 108,
    height = 23)

entry6_img = PhotoImage(file = f"img_textBox6.png")
entry6_bg = canvas.create_image(
    0,0,
    image = entry6_img)

entry6 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry6.place(
    x = 329, y = 541,
    width = 108,
    height = 23)

entry7_img = PhotoImage(file = f"img_textBox7.png")
entry7_bg = canvas.create_image(
   0,0,
    image = entry7_img)

entry7 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry7.place(
    x = 329, y = 579,
    width = 108,
    height = 23)

entry8_img = PhotoImage(file = f"img_textBox8.png")
entry8_bg = canvas.create_image(
    0,0,
    image = entry8_img)

entry8 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry8.place(
    x = 329, y = 617,
    width = 108,
    height = 23)

entry9_img = PhotoImage(file = f"img_textBox9.png")
entry9_bg = canvas.create_image(
    0,0,
    image = entry9_img)

entry9 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry9.place(
    x = 329, y = 656,
    width = 108,
    height = 23)

entry10_img = PhotoImage(file = f"img_textBox10.png")
entry10_bg = canvas.create_image(
    0,0,
    image = entry10_img)

entry10 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry10.place(
    x = 329, y = 692,
    width = 108,
    height = 23)

entry11_img = PhotoImage(file = f"img_textBox11.png")
entry11_bg = canvas.create_image(
    0,0,
    image = entry11_img)

entry11 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry11.place(
    x = 329, y = 731,
    width = 108,
    height = 23)

entry12_img = PhotoImage(file = f"img_textBox12.png")
entry12_bg = canvas.create_image(
    0,0,
    image = entry12_img)

entry12 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry12.place(
    x = 575, y = 316,
    width = 108,
    height = 23)

entry13_img = PhotoImage(file = f"img_textBox13.png")
entry13_bg = canvas.create_image(
    0,0,
    image = entry13_img)

entry13 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry13.place(
    x = 575, y = 354,
    width = 108,
    height = 23)

entry14_img = PhotoImage(file = f"img_textBox14.png")
entry14_bg = canvas.create_image(
   0,0,
    image = entry14_img)

entry14 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry14.place(
    x = 575, y = 392,
    width = 108,
    height = 23)

entry15_img = PhotoImage(file = f"img_textBox15.png")
entry15_bg = canvas.create_image(
    0,0,
    image = entry15_img)

entry15 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry15.place(
    x = 575, y = 431,
    width = 108,
    height = 23)

entry16_img = PhotoImage(file = f"img_textBox16.png")
entry16_bg = canvas.create_image(
    0,0,
    image = entry16_img)

entry16 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry16.place(
    x = 575, y = 467,
    width = 108,
    height = 23)

entry17_img = PhotoImage(file = f"img_textBox17.png")
entry17_bg = canvas.create_image(
    0,0,
    image = entry17_img)

entry17 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry17.place(
    x = 575, y = 506,
    width = 108,
    height = 23)

entry18_img = PhotoImage(file = f"img_textBox18.png")
entry18_bg = canvas.create_image(
    0,0,
    image = entry18_img)

entry18 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry18.place(
    x = 575, y = 542,
    width = 108,
    height = 23)

entry19_img = PhotoImage(file = f"img_textBox19.png")
entry19_bg = canvas.create_image(
    0,0,
    image = entry19_img)

entry19 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry19.place(
    x = 575, y = 580,
    width = 108,
    height = 23)

entry20_img = PhotoImage(file = f"img_textBox20.png")
entry20_bg = canvas.create_image(
   0,0,
    image = entry20_img)

entry20 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry20.place(
    x = 575, y = 618,
    width = 108,
    height = 23)

entry21_img = PhotoImage(file = f"img_textBox21.png")
entry21_bg = canvas.create_image(
    0,0,
    image = entry21_img)

entry21 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry21.place(
    x = 575, y = 657,
    width = 108,
    height = 23)

entry22_img = PhotoImage(file = f"img_textBox22.png")
entry22_bg = canvas.create_image(
   0,0,
    image = entry22_img)

entry22 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry22.place(
    x = 575, y = 693,
    width = 108,
    height = 23)

entry23_img = PhotoImage(file = f"img_textBox23.png")
entry23_bg = canvas.create_image(
    0,0,
    image = entry23_img)

entry23 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry23.place(
    x = 575, y = 732,
    width = 108,
    height = 23)


entry24_img = PhotoImage(file = f"img_textBox24.png")
entry24_bg = canvas.create_image(
    0,0,
    image = entry24_img)

entry24 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry24.place(
    x = 823, y = 317,
    width = 108,
    height = 23)

entry25_img = PhotoImage(file = f"img_textBox25.png")
entry25_bg = canvas.create_image(
    0,0,
    image = entry25_img)

entry25 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry25.place(
    x = 823, y = 355,
    width = 108,
    height = 23)

entry26_img = PhotoImage(file = f"img_textBox26.png")
entry26_bg = canvas.create_image(
    0,0,
    image = entry26_img)

entry26 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry26.place(
    x = 823, y = 393,
    width = 108,
    height = 23)

entry27_img = PhotoImage(file = f"img_textBox27.png")
entry27_bg = canvas.create_image(
   0,0,
    image = entry27_img)

entry27 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry27.place(
    x = 823, y = 432,
    width = 108,
    height = 23)

entry28_img = PhotoImage(file = f"img_textBox28.png")
entry28_bg = canvas.create_image(
    0,0,
    image = entry28_img)

entry28 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry28.place(
    x = 823, y = 468,
    width = 108,
    height = 23)

entry29_img = PhotoImage(file = f"img_textBox29.png")
entry29_bg = canvas.create_image(
   0,0,
    image = entry29_img)

entry29 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry29.place(
    x = 823, y = 507,
    width = 108,
    height = 23)

entry30_img = PhotoImage(file = f"img_textBox30.png")
entry30_bg = canvas.create_image(
    0,0,
    image = entry30_img)

entry30 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry30.place(
    x = 823, y = 543,
    width = 108,
    height = 23)

entry31_img = PhotoImage(file = f"img_textBox31.png")
entry31_bg = canvas.create_image(
   0,0,
    image = entry31_img)

entry31 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry31.place(
    x = 823, y = 581,
    width = 108,
    height = 23)

entry32_img = PhotoImage(file = f"img_textBox32.png")
entry32_bg = canvas.create_image(
    0,0,
    image = entry32_img)

entry32 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry32.place(
    x = 823, y = 619,
    width = 108,
    height = 23)

entry33_img = PhotoImage(file = f"img_textBox33.png")
entry33_bg = canvas.create_image(
    0,0,
    image = entry33_img)

entry33 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry33.place(
    x = 823, y = 658,
    width = 108,
    height = 23)

entry34_img = PhotoImage(file = f"img_textBox34.png")
entry34_bg = canvas.create_image(
    0,0,
    image = entry34_img)

entry34 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry34.place(
    x = 823, y = 694,
    width = 108,
    height = 23)

entry35_img = PhotoImage(file = f"img_textBox35.png")
entry35_bg = canvas.create_image(
    0,0,
    image = entry35_img)

entry35 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry35.place(
    x = 823, y = 733,
    width = 108,
    height = 23)

entry36_img = PhotoImage(file = f"img_textBox36.png")
entry36_bg = canvas.create_image(
    0,0,
    image = entry36_img)

entry36 = Entry(
    font="Batang",
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry36.place(
    x = 880, y = 137,
    width = 110,
    height = 29)

entry37_img = PhotoImage(file = f"img_textBox37.png")
entry37_bg = canvas.create_image(
    0,0,
    image = entry37_img)

entry37 = Entry(
    font="Batang",
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry37.place(
    x = 880, y = 195,
    width = 110,
    height = 29)




window.resizable(False, False)
window.mainloop()
