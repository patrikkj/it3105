from environment import NimEnvironment


configs = {
    
    "nim_params": {
    "N" : 87,
    "K" : 5
    }
}


def main():
    with NimEnvironment(**configs["nim_params"]) as env:
        print("\n\n\n")
        print(" Initial stones:", env.stones)
        mover = 1
        winner = 0
        print("---------------------          GAME START        -----------------------------")
        print("\n")
        while not env.is_finished():
            mover = (mover +1) % 2
            random_move = env.random_move()
            env.move(mover, random_move)
            print (env.players[mover], " , stones removed: ", random_move, " , Stones remaining: ", env.stones)
        
        winner = env.players[env.winner()]
        print("\n")
        print( "====================         GAME FINISHED        ===================")
        print("\n")
        print( "Winner: ", winner)
        print("\n")
        print("CONGRATULATIONS!!!")
        print("\n\n\n")



main()