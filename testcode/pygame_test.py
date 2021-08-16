import pygame
from pygame.locals import *
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("TestGame")
    font = pygame.font.Font(None, 50)

    while True:
        screen.fill((0,0,0))
        text = font.render("This is TEST Script!!", True, (255, 255, 255))
        screen.blit(text, [30, 100])
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__=="__main__":
    main()
