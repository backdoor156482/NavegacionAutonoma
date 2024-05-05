"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def line_reducer(lines, slope_threshold=1):
    grouped_lines = {}  # Usamos un diccionario para agrupar líneas por pendiente
    xbase = {}  # Diccionario para almacenar las coordenadas x base de las líneas agrupadas

    if lines is not None:  # Verifica si hay líneas detectadas
        for index, line in enumerate(lines):  # Itera sobre cada línea detectada junto con su índice
            x1, y1, x2, y2 = line[0]  # Extrae las coordenadas de los puntos extremos de la línea
            m = (y2 - y1) / (x2 - x1)  # Calcula la pendiente de la línea
            if m == 0:  # Si la pendiente es 0 (línea vertical), continúa con la siguiente línea
                continue
            b = y1 - (m*x1)  # Calcula el término independiente de la ecuación de la línea
            xo = (50 - b) / m  # Calcula la intersección de la línea con el eje x en y=50
            rounded_slope = round(m, 1)  # Redondea la pendiente a un decimal

            if index not in grouped_lines:  # Si el índice de la línea no está en ningún grupo existente
                for existing_slope, existing_group in grouped_lines.items():  # Itera sobre los grupos existentes
                    exit_by_distance = False  # Bandera para salir del bucle por distancia
                    if abs(existing_slope - rounded_slope) <= slope_threshold:  # Comprueba si la pendiente está dentro del umbral de tolerancia
                        existing_group.append(index)  # Agrega el índice de la línea al grupo existente
                        xbase[existing_slope].append(xo)  # Agrega la coordenada x base al grupo existente
                        break
                    else:
                        for distance in xbase[existing_slope]:  # Itera sobre las coordenadas x base de las líneas agrupadas
                            if abs(xo - distance) < 100:  # Si la distancia entre las coordenadas x base es menor que 100
                                existing_group.append(index)  # Agrega el índice de la línea al grupo existente
                                xbase[existing_slope].append(xo)  # Agrega la nueva coordenada x base al grupo existente
                                exit_by_distance = True  # Activa la bandera de salida por distancia
                                break
                        if exit_by_distance:  # Si la bandera de salida por distancia está activa, sale del bucle interno
                            break

                else:  # Si no se encontró un grupo existente para la pendiente redondeada
                    grouped_lines[rounded_slope] = [index]  # Crea un nuevo grupo con la pendiente redondeada
                    xbase[rounded_slope] = [xo]  # Agrega la coordenada x base al nuevo grupo
    lines_reduced = list()  # Lista para almacenar las líneas reducidas
    for slope, group_indices in grouped_lines.items():  # Itera sobre los grupos de líneas
        selected_line_index = max(group_indices, key=lambda i: lines[i][0][0]) if slope <= 0 else min(group_indices, key=lambda i: lines[i][0][0])  # Selecciona la línea más a la izquierda o a la derecha según la pendiente
        lines_reduced.append(list([lines[selected_line_index][0]]))  # Agrega la línea seleccionada a la lista de líneas reducidas

    return lines_reduced  # Devuelve la lista de líneas reducidas


vertices = np.array([[(0,720),(100,550),(1180,550),(1280,720)]], dtype=np.int32)  # Se define un array numpy que contiene los vértices de un polígono



def setLanes(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # Aplicar un filtro Gaussiano para reducir el ruido en la imagen
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
    roi_img = np.zeros_like(gray)  # Crear una imagen de ceros del mismo tamaño que la imagen en escala de grises
    cv2.fillPoly(roi_img, vertices, 255)  # Rellenar una región de interés definida por los vértices con el valor 255 (blanco)
    edges = cv2.Canny(gray, 100, 200)  # Detectar bordes en la imagen en escala de grises utilizando el operador de Canny
    roi_img = cv2.bitwise_and(edges, roi_img)  # Aplicar una operación AND bit a bit entre los bordes detectados y la región de interés
    lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, minLineLength=50, maxLineGap=200)  # Detectar líneas en la región de interés utilizando la transformada de Hough probabilística
    lines = line_reducer(lines)  # Reducir el número de líneas detectadas utilizando la función line_reducer
    print(lines)  # Imprimir las líneas detectadas en la consola
    
    # Si se detectaron líneas
    if lines is not None:
        # Iterar sobre todas las líneas detectadas en orden inverso
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]  # Obtener la línea actual
            x1, y1, x2, y2 = line[0]  # Obtener las coordenadas de los puntos extremos de la línea
            m = (y2 - y1) / (x2 - x1)  # Calcular la pendiente de la línea
            # Si la pendiente es menor que 0.1 (casi horizontal), eliminar la línea
            if abs(m) < 0.1:
                lines.pop(i)
                continue
            # Dibujar la línea en la imagen en escala de grises
            cv2.line(gray, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Devolver las líneas detectadas y la imagen en escala de grises con las líneas dibujadas
    return lines, gray


iter_treshold = 0  # Inicializar el contador de iteraciones sin líneas detectadas
turning_cycle_count = 0  # Inicializar el contador de ciclos de giro
threshold_turn_right = 150  # Establecer el umbral para los ciclos de giro hacia la derecha

# Función para calcular el ángulo de dirección del auto
def calculate_steering_angle(lines, image_width):
    
    lines_array = np.array([line[0] for line in lines])  # Convertir las líneas a un array numpy
    global iter_treshold  # Declarar que se utilizará la variable global iter_treshold
    global turning_cycle_count  # Declarar que se utilizará la variable global turning_cycle_count
    
    # Verificar si se detectaron líneas y si el array de líneas no está vacío
    if lines is not None and len(lines_array) > 0:
        iter_treshold = 0  # Reiniciar el contador de iteraciones sin líneas detectadas
        x_mid = np.mean(lines_array[:, 0])  # Calcular el punto medio de todas las líneas detectadas
        deviation = x_mid - (image_width / 2)  # Calcular la desviación del auto respecto al centro
        steering_angle = deviation / (image_width / 2) * 0.5  # Convertir la desviación en un ángulo de dirección
        # Verificar si se deben aplicar ciclos de giro hacia la derecha
        if turning_cycle_count > threshold_turn_right or turning_cycle_count == 0:
            turning_cycle_count = 0  # Reiniciar el contador de ciclos de giro
            return steering_angle  # Devolver el ángulo de dirección calculado
        else:
            turning_cycle_count += 1  # Incrementar el contador de ciclos de giro
            return 0.18  # Si se están acumulando ciclos de giro, mantener el ángulo actual
    else:
        iter_treshold += 1  # Incrementar el contador de iteraciones sin líneas detectadas
        # Verificar si se superó el umbral de iteraciones sin líneas
        if iter_treshold >= 3:
            turning_cycle_count += 1  # Incrementar el contador de ciclos de giro
            return 0.18  # Si no se detectan líneas, mantener el ángulo actual
        else:
            return 0  # Si no se supera el umbral de iteraciones sin líneas, no realizar ningún giro


#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 50

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
    speed = kmh
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        # grey_image = greyscale_cv2(image)
        lanes_image, gris = setLanes(image)

        resized_image = cv2.resize(gris, (600, 338))

        # Calcular ángulo de dirección del auto
        steering_angle = calculate_steering_angle(lanes_image, gris.shape[1])

        display_image(display_img, resized_image)
            
        #update angle and speed
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()