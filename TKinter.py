from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import os

currDir = os.getcwd()
chosenDir = currDir

def aboutUs():
    print("v1.0\n"
          "This software is built by ServeFlow team in Singapore.\n"
          "For any edits, contact Rohit Singh")

def runAnalysis():
    print("Running analysis now. Please do  not press quit.\nThis can take a while :)")

def doNothing():
    print("ok i wont...")

def chooseDir():
    currDir = os.getcwd()
    chosenDir = filedialog.askdirectory(parent = root,initialdir = currDir , title = "Please select a folder")


root = Tk()

initialise = tkinter.messagebox.askquestion("ServeFlow PHA Tool","Welcome to ServeFlow's PHA Data Analysis Tool v1.0\nHope you are doing good\n"
                                                                 "Are you ready to initialise?")
if initialise == "yes":

    photo = PhotoImage(file="flowserve-logo.png")
    label = Label(root, image=photo)
    label.pack()

    menu = Menu(root)
    root.config(menu = menu)

    subMenu = Menu(menu)
    menu.add_cascade(label = "File", menu = subMenu)
    subMenu.add_command(label = "New project...", command = doNothing)
    subMenu.add_command(label = "New...", command = doNothing)
    subMenu.add_separator()
    subMenu.add_command(label = "Exit", command = doNothing)

    editMenu = Menu(menu)
    menu.add_cascade(label = "Help", menu = editMenu)
    editMenu.add_command(label = "About", command = aboutUs)

    # *****ToolBar*****

    toolBar = Frame(root, bg = "blue")

    insertButt = Button(toolBar, text="Choose directory where raw files are kept", command=chooseDir)
    insertButt.pack(side=LEFT, padx=2, pady=2)
    insertButt = Button(toolBar, text = "Run Analysis", command = runAnalysis)
    insertButt.pack(side = LEFT, padx = 2, pady = 2)
    insertPrint = Button(toolBar, text = "Quit", command = toolBar.quit)
    insertPrint.pack(side = RIGHT, padx = 2, pady = 2)

    toolBar.pack(side = TOP, fill = X)

    #******Status Bar******

    statusBar = Label(root, text = "Waiting for instructions", bd = 1, relief = SUNKEN, anchor = W)       #W = west
    statusBar.pack(side = BOTTOM, fill = X)

    chosenDirBar = Label(root, text = "Chosen directory: " +chosenDir, bd = 1, relief = SUNKEN, anchor = W)
    chosenDirBar.pack(side = BOTTOM, fill = X)

    root.mainloop()

if initialise == "no":
    print ("ok no worries. see you next time")

root.mainloop()