from tkinter import *

cam = N

def inputCAM():
    cam = input("input CAM information : ")

win = Tk() # GUI 생성 
win.title("21_2_MAGOK_AI") #상단의 타이틀 지정
win.geometry("500x300") # 크기 설정

fla_greeting = Frame(win)

lab_greeting = Label(fla_greeting, text = "안녕하세요 ~")
lab_greeting.pack()

inputCAM()

if(a != 'Y'):
    inputCAM()

else:
    lab_greeting["text"] = "101호로 안내해드리겠습니다."

    '''
    fla_menuSel = Frame(win)
    lab_greeting.pack_forget()
    lab_menuSel = Label(fla_menuSel, text="메뉴를 선택하세요")
    lab_menuSel.pack()
    fla_menuSel.pack()
    '''

win.mainloop() # GUI가 보이고 종료될때까지 실행함