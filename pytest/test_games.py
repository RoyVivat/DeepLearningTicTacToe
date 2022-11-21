import numpy as np
from games.tictactoe import TicTacToe, RandomTTTPlayer

def test_example_test():
    assert( 1==1 )

def test_tictactoe_init():
    # Testing default board results in empty board
    t = TicTacToe()
    assert( np.all(t.board == np.zeros((3,3))) )
    
    # Testing custom board initialization
    t = TicTacToe({'board': np.ones((3,3))})
    assert( np.all(t.board == np.ones((3,3))))

def test_tictactoe_player_init():
    t = TicTacToe()
    t.init_players( [RandomTTTPlayer("p1", TicTacToe), RandomTTTPlayer("p2", TicTacToe)] )

    assert( t.players[0].name == 'p1' )
    assert( t.players[1].name == 'p2' )

def test_tictactoe_rand_playthrough():
    t = TicTacToe()
    t.init_players( [RandomTTTPlayer("p1", TicTacToe), RandomTTTPlayer("p2", TicTacToe)] )
    t.run()
    assert( t.is_game_over() )

def test_tictactoe_game_history():
    t = TicTacToe()
    t.init_players( [RandomTTTPlayer("p1", TicTacToe), RandomTTTPlayer("p2", TicTacToe)] )
    t.is_saving_history = True
    t.run()
    assert(len(t.game_history) >= 5)
    assert(np.all(t.game_history[0] == np.zeros((3,3))))
