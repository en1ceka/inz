import numpy as np
import pygame

class pgame:
    def __init__(self, board):
        self.board = board

    def set_board(self, board):
        self.board = board

    def draw_board(self):
        pygame.init()
        WIDTH = 800
        HEIGHT = 800
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # board = np.array([[0,-1,0,-1,0,-1,0,-1],
        #                                 [-1,0,-1,0,-1,0,-1,0],
        #                                 [0,-1,0,-1,0,-1,0,-1],
        #                                 [0,0,0,0,-2,0,0,0],
        #                                 [0,0,0,2,0,0,0,0],
        #                                 [1,0,1,0,1,0,1,0],
        #                                 [0,1,0,1,0,1,0,1],
        #                                 [1,0,1,0,1,0,1,0]])

        run = True
        BROWN = (173, 109, 83)
        WHITE = (230,230,230)
        BLACK = (0,0,0)
        CREAM = (255, 229, 204)
        while run:

            for y in range(8):
                for x in range(8):
                    if y%2 == 0:
                        if x%2 == 0:
                            pygame.draw.rect(screen,WHITE,(x*(WIDTH/8),y*(WIDTH/8),WIDTH/8,HEIGHT/8))
                        else:
                            pygame.draw.rect(screen,BROWN,(x*(WIDTH/8),y*(WIDTH/8),WIDTH/8,HEIGHT/8))
                    else:
                        if x%2 == 0:
                            pygame.draw.rect(screen,BROWN,(x*(WIDTH/8),y*(WIDTH/8),WIDTH/8,HEIGHT/8))
                        else:
                            pygame.draw.rect(screen,WHITE,(x*(WIDTH/8),y*(WIDTH/8),WIDTH/8,HEIGHT/8))

            for y in range(8):
                for x in range(8):                
                    if self.board[y][x] == -1:
                        pygame.draw.circle(screen, BLACK, (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/16-5))
                    if self.board[y][x] == 1:
                        pygame.draw.circle(screen, CREAM, (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/16-5))
                    if self.board[y][x] == -2:
                        pygame.draw.circle(screen, BLACK, (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/16-5))
                        pygame.draw.circle(screen, (55,55,55), (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/32-5))
                    if self.board[y][x] == 2:
                        pygame.draw.circle(screen, CREAM, (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/16-5))
                        pygame.draw.circle(screen, (160,160,160), (int(x*(WIDTH/8)) + int(WIDTH/16),int(y*(WIDTH/8)) + int(WIDTH/16)), int(WIDTH/32-5))

        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        run = False
            pygame.display.update()

