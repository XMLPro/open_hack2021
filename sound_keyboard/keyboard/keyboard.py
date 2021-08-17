import pygame
import sys
import os
from sound_keyboard.keyboard.state_controller import (
    KEYMAP,
    KeyboardStateController,
    Direction
)
import cv2

from sound_keyboard.face_gesture_detector.face_gesture_detector import (
    EyeDirection,
    EyeState,
    MouseState,
    detect_gestures,
    Gestures
)

# constants
BACKGROUND_COLOR = (242, 242, 242)
KEYTILE_COLOR = (150, 150, 150)
FONT_COLOR = (12, 9, 10)

FONT_PATH = 'fonts/Noto_Sans_JP/NotoSansJP-Regular.otf'

class Keyboard:
    def __init__(self):
        # initialize app
        pygame.init()
        pygame.display.set_caption('Faceboard')

        # setting initial window size and set window resizable
        self.surface = pygame.display.set_mode((500, 500), pygame.RESIZABLE)

        # setting keyboard controller
        self.keyboard_state_controller = KeyboardStateController()

        self.cap = cv2.VideoCapture(0)
    
    def draw_text(self, char_info):
        char, pos, size = char_info

        font = pygame.font.Font(FONT_PATH, size)
        text = font.render(char, True, FONT_COLOR, None)
        textRect = text.get_rect()
        textRect.center = pos
        self.surface.blit(text, textRect)

    def draw_around(self):

        width = self.surface.get_width()
        height = self.surface.get_height()

        left = (width / 6, height / 2)
        up = (width / 2, height / 6)
        right = (width * 5 / 6, height / 2)
        down = (width / 2, height * 5 / 6)

        center_font_size = int(min(width, height) / 10)
        other_font_size = int(center_font_size * 4 / 10)

        chars = [
            (self.keyboard_state_controller.get_neighbor(Direction.LEFT)[0], left, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.UP)[0], up, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.RIGHT)[0], right, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.DOWN)[0], down, other_font_size),
        ]

        for char in chars:
            self.draw_text(char)
    
    def draw_tile(self, tile_info):
        char, pos, isFocused = tile_info
        pygame.draw.rect(self.surface, KEYTILE_COLOR, pos, isFocused)
        left, top, width, height = pos
        self.draw_text((char, (left + width / 2, top + height / 2), 15))

    def draw_keyboard(self):
        kind = self.keyboard_state_controller.kind
        keymap = KEYMAP[kind]['parent']

        width = self.surface.get_width()
        height = self.surface.get_height()

        cell_size = 1e9
        min_cell_size = 50
        
        row_num = len(keymap)
        col_num = len(keymap[0])

        if min_cell_size * row_num < height:
            cell_size = min_cell_size
        else:
            cell_size = int(height / row_num)

        if min_cell_size * col_num < width:
            cell_size = min(cell_size, min_cell_size)
        else:
            cell_size = min(cell_size, int(width / col_num))
        
        keyboard_width = cell_size * col_num
        keyboard_height = cell_size * row_num

        left = width / 2 - keyboard_width / 2
        top = height / 2 - keyboard_height / 2

        for i, row in enumerate(keymap):
            for j, char in enumerate(row):
                tile_left = left + j * cell_size
                tile_top = top + i * cell_size
                self.draw_tile((char, (tile_left, tile_top, cell_size, cell_size), char == self.keyboard_state_controller.current_parent_char))

    def start(self):

        while True:
            
            self.surface.fill(BACKGROUND_COLOR)

            self.draw_keyboard()
            self.draw_around()

            pygame.display.update()

            gestures: Gestures = Gestures()
            ret, frame = self.cap.read()

            if ret:
                gestures = detect_gestures(frame)
            
            if gestures is None:
                gestures = Gestures()
                gestures.eye_direction = EyeDirection.CENTER
                gestures.left_eye_state = EyeState.OPEN
                gestures.right_eye_state = EyeState.OPEN
                gestures.mouse_state = MouseState.CLOSE

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                
                #TODO(hakomori64) remove it 
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        # self.keyboard_state_controller.move(Direction.LEFT)
                        gestures.eye_direction = EyeDirection.LEFT
                    if event.key == pygame.K_UP:
                        # self.keyboard_state_controller.move(Direction.UP)
                        gestures.eye_direction = EyeDirection.UP
                    if event.key == pygame.K_RIGHT:
                        # self.keyboard_state_controller.move(Direction.RIGHT)
                        gestures.eye_direction = EyeDirection.RIGHT
                    if event.key == pygame.K_DOWN:
                        # self.keyboard_state_controller.move(Direction.DOWN)
                        gestures.eye_direction = EyeDirection.DOWN
                    if event.key == pygame.K_PAGEUP:
                        gestures.left_eye_state = EyeState.CLOSE
                    if event.key == pygame.K_PAGEDOWN:
                        gestures.right_eye_state = EyeState.CLOSE
                    if event.key == pygame.K_RETURN:
                        gestures.mouse_state = MouseState.OPEN
                    


            # Gesturesオブジェクトの状態を読み出して操作を確定する
            direction = gestures.eye_direction
            if direction == EyeDirection.LEFT:
                self.keyboard_state_controller.move(Direction.LEFT)
            elif direction == EyeDirection.UP:
                self.keyboard_state_controller.move(Direction.UP)
            elif direction == EyeDirection.RIGHT:
                self.keyboard_state_controller.move(Direction.RIGHT)
            elif direction == EyeDirection.DOWN:
                self.keyboard_state_controller.move(Direction.DOWN)

if __name__ == '__main__':
    Keyboard().start()