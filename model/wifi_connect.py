import socket
import numpy as np
from model import GestureClassifier

# ESP32's IP address (check the Serial Monitor for the IP address when ESP32 connects to WiFi)
ESP_IP = '172.20.10.5'
ESP_PORT = 80

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ESP_IP, ESP_PORT))

label = ['fist', 'pinch2finger', 'pinch3finger', 'release2finger', 'release3finger']

# Send a request (optional)
s.sendall(b'Hello ESP32!')
#
model = GestureClassifier()
model.load_model("gestureXR_model.keras")

try:
    while True:
        input_data = []
        data = s.recv(1024)  # Receive data from the server (max 1024byte)
        if not data:
            break

        # Decode the byte string to a regular string
        decoded_data = data.decode('utf-8')
        lines = decoded_data.strip().split("\r\n")
        try:
            # Split the line by commas
            string_numbers = lines[0].split(',')
            # Convert each string to a float and append to the list
            float_numbers = [float(num) for num in string_numbers]
        except ValueError:
            float_numbers = -1
        print(float_numbers)
        if not isinstance(float_numbers, int) and len(float_numbers) == 12:
            numbers = np.asarray(float_numbers)
            numbers = numbers.reshape(1, -1)  # Reshape to add a batch dimension
            print(numbers)
            prediction = model.model.predict(numbers, verbose=0)
            print(label[np.argmax(prediction)])
        else:
            print("Reading error")

finally:
    s.close()
    print('Connection closed')

import pygame
import sys
from random import randrange

pygame.init()

# window size
screen_width = 1200
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Menu")

# colors
WHITE = (255, 255, 255)
BLACK = (randrange(255), randrange(255), randrange(255))
RED = (randrange(255), randrange(255), randrange(255))

# size of the square and distance between each one
menu_item_width = 60
menu_item_height = 120
menu_spacing = 50

# square list
menu_items = []
for i in range(10):
    menu_item_x = i * (menu_item_width + menu_spacing) + 100
    menu_item_y = (screen_height - menu_item_height) // 2
    menu_items.append(pygame.Rect(menu_item_x, menu_item_y, menu_item_width, menu_item_height))

selected_item = menu_items[5]  # selected square("menu")

# max min size of squares
min_item_width = 20
min_item_height = 40
max_item_width = 120
max_item_height = 240


def move_selected_item(direction):
    global selected_item
    if selected_item:
        index = menu_items.index(selected_item)
        if direction == 'left' and index > 0:
            selected_item = menu_items[index - 1]
        elif direction == 'right' and index < len(menu_items) - 1:
            selected_item = menu_items[index + 1]

def draw_menu():
    screen.fill(WHITE)  # clean everything

    # draw the square
    for item in menu_items:
        color = BLACK
        if item == selected_item:
            color = RED  # If it is a selected menu item, mark it in red

        pygame.draw.rect(screen, color, item)  # draw the square

    pygame.display.flip()  # update the new square

def zoom_menu(zoom_factor):
    global selected_item
    if selected_item:
        # obtain the central
        center_x = selected_item.x + selected_item.width // 2
        center_y = selected_item.y + selected_item.height // 2

        # Calculate the scaled size
        new_width = selected_item.width + zoom_factor
        new_height = selected_item.height + zoom_factor

        # Make sure the size is within the minimum and maximum values
        new_width = max(min(new_width, max_item_width), min_item_width)
        new_height = max(min(new_height, max_item_height), min_item_height)

        # Update the menu item size and keep the center position unchanged
        selected_item.width = new_width
        selected_item.height = new_height
        selected_item.x = center_x - new_width // 2
        selected_item.y = center_y - new_height // 2


def reset_menu():
    global menu_items, selected_item
    # Reset menu item size and position
    menu_items = []
    for i in range(10):
        menu_item_x = i * (menu_item_width + menu_spacing) + 100
        menu_item_y = (screen_height - menu_item_height) // 2
        menu_items.append(pygame.Rect(menu_item_x, menu_item_y, menu_item_width, menu_item_height))
    selected_item = menu_items[5]


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left click
                move_selected_item('left')

            elif event.button == 3:  # right click
                move_selected_item('right')

        elif event.type == pygame.KEYDOWN:
            if selected_item:
                if event.key == pygame.K_1:  # Press keyboard 1 to enlarge the selected menu item
                    zoom_menu(10)
                elif event.key == pygame.K_2:  # Press keyboard 2 to shrink the selected menu item
                    zoom_menu(-10)
            if event.key == pygame.K_q:  # Press button q to initialize all changes
                reset_menu()

    draw_menu()  # Draw menu bar interface

pygame.quit()  # Exit Pygame
