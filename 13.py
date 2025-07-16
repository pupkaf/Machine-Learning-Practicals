import tkinter as tk
import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player  # Switch player

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:
            return self.board[0, 0]
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:
            return self.board[0, 2]
        return 0

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.game = TicTacToe()
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.master, text='', font='Arial 20', width=5, height=2,
                                   command=lambda row=i, col=j: self.on_button_click(row, col))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button
        self.reset_button = tk.Button(self.master, text='Reset', command=self.reset_game)
        self.reset_button.grid(row=3, column=0, columnspan=3)

    def on_button_click(self, row, col):
        if self.game.board[row, col] == 0:
            self.game.make_move(row, col)
            self.buttons[row][col].config(text='X' if self.game.current_player == 2 else 'O')
            winner = self.game.check_winner()
            if winner != 0:
                self.show_winner(winner)
            elif np.all(self.game.board != 0):
                self.show_winner(None)

    def show_winner(self, winner):
        if winner:
            message = f'Player {int(winner)} wins!'
        else:
            message = 'It\'s a draw!'
        tk.messagebox.showinfo("Game Over", message)
        self.reset_game()

    def reset_game(self):
        self.game.reset()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='')

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Крестики-нолики")
    app = TicTacToeGUI(root)
    root.mainloop()
