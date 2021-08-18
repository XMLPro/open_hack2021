import pygame
import sys
import cv2
import time
from utils import (
    convert_eye_direction_to_direction
)
from sound_keyboard.keyboard.state_controller import (
    KEYMAP,
    KeyboardStateController,
    Direction
)

from sound_keyboard.face_gesture_detector.face_gesture_detector import (
    EyeDirection,
    EyeState,
    MouseState,
    detect_gestures,
    Gestures
)

# constants
BACKGROUND_COLOR = (242, 242, 242)
KEYTILE_COLOR = (220, 220, 220)
OVERLAY_COLOR = (0, 0, 0, 180)
FONT_COLOR = (12, 9, 10)
MAX_DELAY = 0.1

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

        # state
        self.previous_gestures = None
        self.delay = 0
    
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

        left = (width / 8, height / 2)
        up = (width / 2, height / 8)
        right = (width * 7 / 8, height / 2)
        down = (width / 2, height * 7 / 8)

        center_font_size = int(min(width, height) / 12)
        other_font_size = int(center_font_size * 4 / 12)

        chars = [
            (self.keyboard_state_controller.get_neighbor(Direction.LEFT)[0], left, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.UP)[0], up, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.RIGHT)[0], right, other_font_size),
            (self.keyboard_state_controller.get_neighbor(Direction.DOWN)[0], down, other_font_size),
        ]

        for char in chars:
            self.draw_text(char)
    
    def draw_tile(self, char, center, radius, tile_color, border_size):
        pygame.draw.circle(self.surface, tile_color, center, radius, border_size)
        self.draw_text((char, center, 15))

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
        
        padding = 5
        
        keyboard_width = cell_size * col_num + padding * (col_num - 1)
        keyboard_height = cell_size * row_num + padding * (row_num - 1)

        left = width / 2 - keyboard_width / 2
        top = height / 2 - keyboard_height / 2

        for i, row in enumerate(keymap):
            for j, char in enumerate(row):
                x = left + j * cell_size + (padding * j) + cell_size // 2
                y = top + i * cell_size + (padding * i) + cell_size // 2
                is_focused = char == self.keyboard_state_controller.current_parent_char
                # self.draw_tile(
                #    (char, (x, y), cell_size / 2, is_focused))
                self.draw_tile(
                    char,
                    (x, y),
                    cell_size / 2,
                    KEYTILE_COLOR if is_focused else BACKGROUND_COLOR,
                    3 if is_focused else 1
                )
        
        # draw currently selected text
        self.draw_text((self.keyboard_state_controller.text, (width / 2, top - 10), 20))

    def updateKeyboardState(self, gestures: Gestures):

        # Gesturesオブジェクトの状態を読み出して操作を確定する

        if gestures.eye_direction != EyeDirection.CENTER:
            direction = convert_eye_direction_to_direction(gestures.eye_direction)
            self.keyboard_state_controller.move(direction)
            return True
        
        if gestures.left_eye_state == EyeState.CLOSE and gestures.right_eye_state == EyeState.OPEN:
            # back
            self.keyboard_state_controller.back()
            return True
        
        if (self.previous_gestures is None or self.previous_gestures.mouse_state == MouseState.CLOSE) and gestures.mouse_state == MouseState.OPEN:
            # select
            self.keyboard_state_controller.select()
            return True
        
        return False
    
    def draw_child_keyboard(self):
        
        width = self.surface.get_width()
        height = self.surface.get_height()

        cell_size = int(min(width, height) / 10)

        center_char = self.keyboard_state_controller.current_parent_char
        current_char = self.keyboard_state_controller.current_child_char

        padding = 5
        self.draw_tile(
            center_char,
            (width / 2, height / 2),
            cell_size / 2,
            BACKGROUND_COLOR if center_char == current_char else KEYTILE_COLOR,
            0 if center_char == current_char else 1
        )
        
        for direction in Direction:

            char = self.keyboard_state_controller.get_child_char(center_char, direction)
            x, y = direction.value
            self.draw_tile(
                char,
                (width / 2 + (cell_size + padding) * x, height / 2 + (cell_size + padding) * y),
                cell_size / 2,
                BACKGROUND_COLOR if char == current_char else KEYTILE_COLOR,
                0 if char == current_char else 1
            )

    
    def draw(self):

        # show parent view
        self.surface.fill(BACKGROUND_COLOR)

        self.draw_keyboard()
        self.draw_around()
        
        if self.keyboard_state_controller.selected_parent:
            # show overlay
            self.overlay = pygame.Surface([self.surface.get_width(), self.surface.get_height()], pygame.SRCALPHA, 32)
            self.overlay.convert_alpha()
            self.overlay.fill(OVERLAY_COLOR)
            self.surface.blit(self.overlay, (0, 0))
            self.draw_child_keyboard()

        
        pygame.display.update()
    
    def update(self):

        gestures: Gestures = Gestures()
        ret, frame = self.cap.read()

        if not ret:
            return

        gestures = detect_gestures(frame)

        # for debug
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
            
        #TODO(hakomori64) remove it 
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            # self.keyboard_state_controller.move(Direction.LEFT)
            gestures.eye_direction = EyeDirection.LEFT
        if keys[pygame.K_UP]:
            # self.keyboard_state_controller.move(Direction.UP)
            gestures.eye_direction = EyeDirection.UP
        if keys[pygame.K_RIGHT]:
            # self.keyboard_state_controller.move(Direction.RIGHT)
            gestures.eye_direction = EyeDirection.RIGHT
        if keys[pygame.K_DOWN]:
            # self.keyboard_state_controller.move(Direction.DOWN)
            gestures.eye_direction = EyeDirection.DOWN
        if keys[pygame.K_PAGEUP]:
            gestures.left_eye_state = EyeState.CLOSE
        if keys[pygame.K_PAGEDOWN]:
            gestures.right_eye_state = EyeState.CLOSE
        if keys[pygame.K_RETURN]:
            gestures.mouse_state = MouseState.OPEN
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
        
        state_updated = False
        if self.delay <= 0:
            state_updated = self.updateKeyboardState(gestures)
        
        return state_updated
        

    def start(self):

        while True:
            
            start = time.time()

            self.draw()
            state_updated = self.update()

            frame_time = time.time() - start
            if state_updated:
                self.delay = MAX_DELAY
            else:
                self.delay = max(self.delay - frame_time, 0)


if __name__ == '__main__':
    Keyboard().start()