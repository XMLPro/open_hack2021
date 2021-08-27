import pygame
import sys
import cv2
import time
from utils import (
    convert_eye_direction_to_direction
)
from sound_keyboard.queue import (
    get_queue
)
from sound_keyboard.keyboard.state_controller import (
    KEYMAP,
    KeyboardStateController,
    Direction
)
from sound_keyboard.sound.sound import (
    read_aloud
)

from sound_keyboard.face_gesture_detector.face_gesture_detector import (
    EyeDirection,
    EyeState,
    MouthState,
    Gestures
)

# constants
BACKGROUND_COLOR = (242, 242, 242)
KEYTILE_COLOR = (242, 242, 242)
OVERLAY_COLOR = (0, 0, 0, 180)
FONT_COLOR = (12, 9, 10)
MAX_DELAY = 0.1

FONT_PATH = 'fonts/Noto_Sans_JP/NotoSansJP-Regular.otf'

class Keyboard:
    def __init__(self, queue):

        self.queue = queue
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

    def draw_tile(self, char, center, radius, tile_color, border_size, font_size = 15):
        pygame.draw.circle(self.surface, tile_color, center, radius, border_size)
        self.draw_text((char, center, font_size))

    def draw_keyboard(self):
        kind = self.keyboard_state_controller.kind
        keymap = KEYMAP[kind]['parent'][0]

        width = self.surface.get_width()
        height = self.surface.get_height()

        base = width // 2

        cell_sizes = [70, 40, 30]
        font_sizes = [70, 40, 30]
        distances = [0, base // 2, base * 4 // 5]

        
        padding = 5

        center_index = keymap.index(self.keyboard_state_controller.current_parent_char)

        for dir in range(-2, 3):
            index = center_index + dir
            cell_size = cell_sizes[abs(dir)]
            font_size = font_sizes[abs(dir)]
            distance = distances[abs(dir)]

            sign = 1 if dir > 0 else -1

            if 0 <= index < len(keymap):
                self.draw_tile(
                    keymap[index] if abs(dir) != 2 else '...',
                    (width // 2 +  sign * distance, height // 2),
                    cell_size,
                    KEYTILE_COLOR,
                    0,
                    font_size
                )
        
        # draw currently selected text
        self.draw_text((self.keyboard_state_controller.text, (width / 2, height * 7 // 8), 20))

    def updateKeyboardState(self, gestures: Gestures):

        # Gesturesオブジェクトの状態を読み出して操作を確定する

        if gestures.eye_direction != EyeDirection.CENTER:
            direction = convert_eye_direction_to_direction(gestures.eye_direction)
            self.keyboard_state_controller.move(direction)
            return True
        
        if (self.previous_gestures is None or self.previous_gestures.left_eye_state == EyeState.OPEN) and gestures.left_eye_state == EyeState.CLOSE:
            # back
            self.keyboard_state_controller.back()
            return True
        
        if (self.previous_gestures is None or self.previous_gestures.right_eye_state == EyeState.OPEN) and gestures.right_eye_state == EyeState.CLOSE:
            # select
            self.keyboard_state_controller.select()
            return True
        
        if (self.previous_gestures is None or self.previous_gestures.mouth_state == MouthState.CLOSE) and gestures.mouth_state == MouthState.OPEN:
            if self.keyboard_state_controller.text != "":
                read_aloud(self.keyboard_state_controller.text)
            self.keyboard_state_controller.clear()
            return True
        
        return False
    
    def draw_child_keyboard(self):

        kind = self.keyboard_state_controller.kind
        keymap = KEYMAP[kind]['children'][self.keyboard_state_controller.current_parent_char]
        width = self.surface.get_width()
        height = self.surface.get_height()

        base = width // 2

        cell_sizes = [70, 40, 30]
        font_sizes = [60, 30, 20]
        distances = [0, base // 2, base * 4 // 5]

        center_index = keymap.index(self.keyboard_state_controller.current_child_char)

        for dir in range(-2, 3):
            index = center_index + dir
            cell_size = cell_sizes[abs(dir)]
            font_size = font_sizes[abs(dir)]
            distance = distances[abs(dir)]

            sign = 1 if dir > 0 else -1

            if 0 <= index < len(keymap):
                self.draw_tile(
                    keymap[index] if abs(dir) != 2 else '...',
                    (width // 2 +  sign * distance, height // 3),
                    cell_size,
                    KEYTILE_COLOR,
                    0,
                    font_size
                )
        

    def draw(self):

        # show parent view
        self.surface.fill(BACKGROUND_COLOR)

        if self.keyboard_state_controller.selected_parent:
            self.draw_child_keyboard()
        else:
            self.draw_keyboard()
            
        
        pygame.display.update()
    
    def update(self):

        gestures: Gestures = None
        while not self.queue.empty():
            g, enqueued_at = self.queue.get()
            now = time.time()
            # print('received gestures enqueued at: ', enqueued_at, 'now: ', now)
            if now - enqueued_at <= 0.3:
                gestures = g
                break

        # for debug
        if gestures is None:
            gestures = Gestures(
                eye_direction = EyeDirection.CENTER,
                left_eye_state = EyeState.OPEN,
                right_eye_state = EyeState.OPEN,
                mouth_state = MouthState.CLOSE,
            )

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
            gestures.mouth_state = MouthState.OPEN
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
        
        state_updated = False
        if self.delay <= 0:
            state_updated = self.updateKeyboardState(gestures)
        
        self.previous_gestures = gestures
        
        return state_updated
        
    def run(self):

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
    queue = get_queue()
    Keyboard(queue).run()