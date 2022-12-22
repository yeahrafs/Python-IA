from tkinter import *
from tkinter import filedialog
import tkinter
from PySimpleGUI import PySimpleGUI as sg
from PIL import Image, ImageTk
import imutils
import numpy as np
from typing import Text
import cv2 as cv
import numpy as np
import face_recognition as fr

#Define a função onde serão colocas as imagens
def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto) #Carrega a imagem
    rostos = fr.face_encodings(foto) #Reconhece o rosto
    #Se encontrar rostos, retorna TRUE, se não, FALSE
    if(len(rostos) > 0): #Tamanho
        return True, rostos
    else:
        return False, []

#Define a função para pegar os rostos
def get_rostos():
    #Cria uma lista para guardar os rostos conhecidos e o nome do rosto encontrado
    rostos_conhecidos = []
    nomes_dos_rostos = []

    #Pega a função reconhece_face e coloca a imagem
    rafaela = reconhece_face("Fotos Trabalho\Foto1.jpeg") 
    
    if(rafaela[0]):
        rostos_conhecidos.append(rafaela[1][0]) #Array - Pegou o tamanho no len, e nessa imagem tem apenas um rosto
        nomes_dos_rostos.append("Rafaela") #Nome em que ficará na legenda do rosto

    rafaela = reconhece_face("C:\\Users\\rafae\\Desktop\\Fotos Trabalho\\Foto2.jpeg")
    if(rafaela[0]):
        rostos_conhecidos.append(rafaela[1][0])
        nomes_dos_rostos.append("Rafaela")

    rafaela = reconhece_face("C:\\Users\\rafae\\Desktop\\Fotos Trabalho\\Foto3.jpeg")
    if(rafaela[0]):
        rostos_conhecidos.append(rafaela[1][0])
        nomes_dos_rostos.append("Rafaela")
    
    rafaela = reconhece_face("C:\\Users\\rafae\\Desktop\\Fotos Trabalho\\Foto4.jpeg")
    if(rafaela[0]):
        rostos_conhecidos.append(rafaela[1][0])
        nomes_dos_rostos.append("Rafaela")
    
    return rostos_conhecidos, nomes_dos_rostos



def visualizar():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            rostos_conhecidos, nomes_dos_rostos = get_rostos()

    #Para reconheceimento facial, será utilizado o HaarCascade
            xml_haar_cascade = ('C:\\Users\\rafae\\Desktop\\Prog. orientada a objetos e dados\\PROJETO\\haarcascade_frontalface_alt2.xml')

    #Carregar Classificador
            faceClassifier = cv.CascadeClassifier(xml_haar_cascade) 

    #Abrir Camera
            capture = cv.VideoCapture(0)
            capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                ret, frame_color = capture.read()
                rgb_frame = frame_color[:, :, ::-1]
                
                gray = cv.cvtColor(frame_color, cv.COLOR_BGR2GRAY)

                localizacao_dos_rostos = fr.face_locations(rgb_frame)
                rosto_desconhecido = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

                rosto = faceClassifier.detectMultiScale(gray)

                #Moldura em volta do rosto
                for (cima, dir, baixo, esq) in rosto:                                                                              
                    
                    for (cima, dir, baixo, esq), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecido):
                        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
                        print(resultados)

                        face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)
                
                        melhor_id = np.argmin(face_distances)
                    
                        if resultados[melhor_id]:
                            nome = nomes_dos_rostos[melhor_id]
                            cv.rectangle(frame_color, (esq, cima -35), (dir, baixo), (106, 255, 106), 3)
                            cv.putText(frame_color, nome, (esq + 6, baixo- 6), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        else:
                            nome = "Desconhecido"
                            cv.rectangle(frame_color, (esq, cima -35), (dir, baixo), (0, 0, 255), 3)
                            cv.putText(frame_color, nome, (esq + 6, baixo- 6), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)


                cv.imshow('Camera', frame_color)
            
            #Para sair da câmera:
                if cv.waitKey(1) == ord("s"):
                    break        

        else:
            lblVideo.image = ""
            cap.release()

def iniciar():
    global cap
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    visualizar()

def fechar():
    global cap 
    cap.release()


cap = None
root = Tk()


btnIniciar = Button(root, text="Visualizar", width=45, command=iniciar )
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = Button(root, text="Fechar", width=45, command=fechar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)

root.title('Você conhece esta pessoa?')
root.mainloop()