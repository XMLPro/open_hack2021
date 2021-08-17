import pygame
import sys
import os
from sound_keyboard.keyboard.state_controller import (
    KeyboardStateController,
    Direction
)

# constants
BACKGROUND_COLOR = (242, 242, 242)
FONT_COLOR = (12, 9, 10)

FONT_PATH = 'fonts/Noto_Sans_JP/NotoSansJP-Regular.otf'

class Keyboard:
    def __init__(self):
        # initialize app
        pygame.init()
        pygame.display.set_caption('Smile!')

        # setting initial window size and set window resizable
        self.surface = pygame.display.set_mode((500, 500), pygame.RESIZABLE)

        # setting keyboard controller
        self.keyboard_state_controller = KeyboardStateController()


    def start(self):

        while True:
            
            self.surface.fill(BACKGROUND_COLOR)

            font = pygame.font.Font(FONT_PATH, int(min(self.surface.get_width(), self.surface.get_height()) / 10))
            text = font.render(self.keyboard_state_controller.current_parent_char, True, FONT_COLOR, None)
            textRect = text.get_rect()
            textRect.center = (self.surface.get_width() / 2, self.surface.get_height() / 2)

            self.surface.blit(text, textRect)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                
                #TODO(hakomori64) replace current char using gesture
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.keyboard_state_controller.move(Direction.LEFT)
                    if event.key == pygame.K_UP:
                        self.keyboard_state_controller.move(Direction.UP)
                    if event.key == pygame.K_RIGHT:
                        self.keyboard_state_controller.move(Direction.RIGHT)
                    if event.key == pygame.K_DOWN:
                        self.keyboard_state_controller.move(Direction.DOWN)
                    

if __name__ == '__main__':
    print (os.getcwd())
    Keyboard().start()