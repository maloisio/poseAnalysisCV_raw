import tensorflow as tf
import numpy as np
import cv2 as cv
import os


diretorio_atual = os.path.dirname(os.path.realpath(__file__))
caminho_modelo = os.path.join(diretorio_atual, 'models', 'lite-model_movenet_singlepose_lightning_3.tflite')
interpreter = tf.lite.Interpreter(model_path=caminho_modelo)

interpreter.allocate_tensors()
cap = cv.VideoCapture("video_concatenado.mp4") #coloque aqui o nome do video

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

class Individuo:
    def  __init__(self, genero):
        self.genero = genero
        self.estado = False
        self.reto = False
        self.estado_anterior = 0
        self.contador_de_flexoes_idividual = 0
        self.estado_pre_anterior = 0
    
    def verificar_estado(self,keypoints):
        if(self.estado != 0 or self.estado != self.estado_anterior):
            self.estado_pre_anterior = self.estado_anterior
            self.estado_anterior = self.estado
        for linha in keypoints_with_scores[0]:
            if(not(linha[12][0] > linha[8][0]) and (linha[11][0] > linha[6][0])):
                self.estado = str("estendido")
            else:
                self.estado = str("flexionado")

    def confere_mov(self, tempo, contflec,contelap):
        if (tempo > contelap + 20):
            if(self.estado_pre_anterior == "estendido" and self.estado_anterior == "flexionado" and self.estado == "estendido"):
                    self.contador_de_flexoes_idividual = self.contador_de_flexoes_idividual + 1
                    print("flexao")
                    contflec = contflec + 1
                    contelap = tempo
        vetor_de_resposta = [contflec,contelap]
        return vetor_de_resposta
    
def mostra_posicoes(keypoints):
    for linha in keypoints_with_scores[0]:
            print("olhos",[linha[0]])
            # print("ombro esquerdo",[linha[5]])
            print("ombro direito",[linha[6]])
            # print("cotovelo esquerdo",[linha[7]])
            print("cotovelo direito",[linha[8]])
            # print("mao esquerdo",[linha[9]])
            # print("mao direito",[linha[10]])
            print("quadril esquerdo",[linha[11]])
            print("quadril direito",[linha[12]])
            # print("joelho esquerdo",[linha[13]])
            # print("joelho direito",[linha[14]])
            # print("pe esquerdo",[linha[15]])
            # print("pe direito",[linha[16]])


def erro_de_10(valor1,valor2):
    if(valor1 > valor2):
        if(valor1 < valor2 * 1.15):
            return True
        else:
            return False
    else:
        if(valor1*1.15 > valor2):
            return True
        else:
            return False
        

def verificar_reto(keypoints):
    for linha in keypoints_with_scores[0]:
        if(erro_de_10(float(linha[12][1])/float(linha[12][0]),float(linha[6][1])/float(linha[6][0])) and erro_de_10(float(linha[11][1])/float(linha[11][0]),float(linha[5][1])/float(linha[5][0]))):
            return True
        else:
            return False
        

def check_flexao(keypoints, tempo, contflec,contelap):
        #print(tempo,contelap)
        if (tempo > contelap + 20):
            for linha in keypoints_with_scores[0]:
                if(linha[12][0] > linha[8][0]) and (linha[11][0] > linha[6][0]) : #talvez uma lógica para ver se está tendo flexão
                    print(float(linha[6][1]))
                    contflec = contflec + 1
                    contelap = tempo
                    
        vetor_de_resposta = [contflec,contelap]
        
        return vetor_de_resposta
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            cv.circle(frame2, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def verifica_estado(keypoints):
    return 1
Pessoa = Individuo("Masculino")
contagem_de_vezes = 0
limitador_de_frames = int(input("Defina limitador de quantidade total de frames:"))
contador_De_flexoes = contagem_de_vezes
contador_de_tempo_elapsado = contagem_de_vezes
res = (0,0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #frame = cv.rotate(frame, cv.ROTATE_180);
    
    frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta

    #reshape imagem para 192x192x3
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)

    #inputs e ouputs
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #predicoes e pontos
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    # if contagem_de_vezes%10 == 0: #A cada 10 frames me printa a posição de cada coisa
    #     mostra_posicoes(keypoints_with_scores)
            
    # if(verificar_reto(keypoints_with_scores)):
    #     res = check_flexao(keypoints_with_scores, contagem_de_vezes, contador_De_flexoes, contador_de_tempo_elapsado)   
    
    # contador_De_flexoes = res[0]
    # contador_de_tempo_elapsado = res[1]
    contagem_de_vezes = contagem_de_vezes + 1
    Pessoa.verificar_estado(keypoints_with_scores)
    res =  Pessoa.confere_mov(contagem_de_vezes, contador_De_flexoes, contador_de_tempo_elapsado)
    contador_De_flexoes = res[0]
    contador_de_tempo_elapsado = res[1]
    if contagem_de_vezes > limitador_de_frames: #limitador de quantos frames do video ver
        print("Parar e sair")
        exit()
    # print(contagem_de_vezes)
    #desenha o frame com os pontos
    draw_connections(frame, keypoints_with_scores, EDGES, 0.2)
    draw_keypoints(frame, keypoints_with_scores, 0.2)

    frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
    frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
    cv.imshow("tela2", frame2)
    cv.imshow("tela", frame)
    # print("contador de flexoes",contador_De_flexoes)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
print(contagem_de_vezes)
print(contador_De_flexoes)
cap.release()
cv.destroyWindow()