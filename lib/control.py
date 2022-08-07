def allPeersConnectedControlBlock():
    print('All the peers have been connected? [y/n]:')
    userInput = input()
    if(userInput != 'y' and userInput != 'n'):
        print('Not a possible value, please choose "y" for YES and "n" for NO')
    if(userInput != 'y'):
        allPeersConnectedControlBlock()
    else:
        print("Starting distributed model training...")
