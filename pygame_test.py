import pygame
import numpy as np

import os
import glob
import tensorflow as tf
import multiprocessing
import ctypes

from games.game import Turn
from games.tictactoe import TicTacToe
from players.az_multiproc import AlphaZeroPlayer

WIDTH, HEIGHT = 600, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

BG_COLOR = ((230, 250, 250)) # Background color
BLACK = ((0, 0, 0))
FPS = 30

def draw_window(turn, board):
    WIN.fill((230, 250, 250))
    draw_squares(board)
    draw_ttt_board()
    draw_mouse(turn)

def draw_squares(az_pred, board, clicking):
    for i in range(3):
        for j in range(3):
            if board[i,j] != 0:
                continue

            s = pygame.Surface((600,600))
            
            if az_pred[i,j] > 0.5:
                val = az_pred[i,j]*2-1
                s.set_alpha(int(val*255))
                pygame.draw.rect(s,pygame.Color(0,255,0),(200*j,200*i,200,200))
            else:
                val = (1-az_pred[i,j])*2-1
                s.set_alpha(int(val*255))
                pygame.draw.rect(s,pygame.Color(255,0,0),(200*j,200*i,200,200))

            WIN.blit(s, (0,0))

    for i in range(3):
        for j in range(3):
            if board[i,j] == 1:
                draw_x(100 + 200*j, 100 + 200*i, BLACK)
            if board[i,j] == -1:
                draw_o(100 + 200*j, 100 + 200*i, BLACK)

    if clicking:
        pass


def draw_ttt_board():
    pygame.draw.rect(WIN,BLACK,(198,0,4,600))
    pygame.draw.rect(WIN,BLACK,(398,0,4,600))
    pygame.draw.rect(WIN,BLACK,(0,198,600,4))
    pygame.draw.rect(WIN,BLACK,(0,398,600,4))

def draw_mouse(turn):
    mouse_x, mouse_y = pygame.mouse.get_pos()

    if turn == Turn.P1:
        draw_x(mouse_x, mouse_y, BLACK)
    else:
        draw_o(mouse_x, mouse_y, BLACK)

def draw_x(x, y, color):
    pygame.draw.line(WIN,color,(x-80, y-80),(x+80, y+80), 10)
    pygame.draw.line(WIN,color,(x-80, y+80),(x+80, y-80), 10)

def draw_o(x, y, color):
    pygame.draw.circle(WIN, color, (x, y), 80, 10)

def handle_clicking(clicking, board):
    if clicking:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = round((mouse_x-100)/200)
        y = round((mouse_y-100)/200)
        if board[y,x] == 0:
            pygame.draw.circle(WIN, BLACK, (x*200+100, y*200+100), 10, 10)

def main():
    clock = pygame.time.Clock()
    run = True

    turn = Turn.P1
    board = np.zeros((3,3))

    clicking = False

    folder_path = os.getcwd() + "/saved_models/"
    file_type = r'/*'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    model = tf.keras.models.load_model(max_file)

    az_player = AlphaZeroPlayer('p', TicTacToe, model)

    t = TicTacToe()
    
    shared_mem = multiprocessing.Array(ctypes.c_float, 9)
    proc = multiprocessing.Process(target=az_player.play, args=[t.get_game_state(), shared_mem])
    proc.start()

    while run:


        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                     clicking = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    clicking = False
                    
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    x = round((mouse_x-100)/200)
                    y = round((mouse_y-100)/200)
                    board[y,x] = turn

                    t.update_game_state((y,x))
                    proc.kill()

                    turn = Turn.next(turn)

        if not proc.is_alive():
            proc = multiprocessing.Process(target=az_player.play, args=[t.get_game_state(), shared_mem])
            proc.start()

        az_pred = np.frombuffer(shared_mem.get_obj(), dtype=ctypes.c_float).reshape((3,3))

        # Draw Window ------------------------------------------------
        WIN.fill(BG_COLOR)
        draw_squares(az_pred, board, clicking)
        draw_ttt_board()
        draw_mouse(turn)
        handle_clicking(clicking, board)

        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    main()