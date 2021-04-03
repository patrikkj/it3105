from dataclasses import dataclass, field

import numpy as np

@dataclass
class MCNode:
    def __init__(self, state):
        self.state = state
        #Var ikkje init metode her, visste ikkje om det var med vilje eller ikkje.  La til state
        player: int = 0
        reward: int = 0
        visit_count: int = 0
        is_leaf: bool = False
        successors: dict = field(default_factory=dict)


class MCTree:
    def __init__(self, root, tree_policy=1, target_policy):
        self.root = root
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy)
        ...

    def search(self, env):
        """ (1) - Tree Search
        Traversing the tree from the root to a leaf node by using the tree policy.
        """
        search_game = env.copy()
        cur_node = self.root            
        while not cur_node.is_leaf: 
            # assuming that tree policy is the c constant in the uct formula
            best_action= np.argmax([self.uct(cur_node, action, self.tree_policy) for action in cur_node])            
            search_game.move(best_action, cur_node.player) #Riktig? er player i ein node player som har neste move ut av denna nodens state?
            cur_node = cur_node.successors[best_action]

        return cur_node, search_game


    def node_expansion(self, node, search_game):
        """ (2)
        Generating some or all child states of a parent state, and then connecting the tree node housing
        the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
        """
        actions = search_game.get_legal_actions()
        for action in actions:
            game = search_game.copy()
            game.move(action, node.player) #riktig player?
            child = MCNode(game.get_observation()) # Vi må vel sette player når vi initialiserer noder? Den hentes vel ikkje fra get_observation
            node.successors[action] = child

        return node

        
        ...

    def leaf_evaluation(self, node):
        """ (3)
        Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
        policy from the leaf node’s state to a final state.
        """
        ...

    def backpropagate(self, node, score):
        """ (4)
        Passing the evaluation of a final state back up the tree, updating relevant data (see course
        lecture notes) at all nodes and edges on the path from the final state to the tree root.
        """
        while node is not None:
            node.score += score
            node.visit_count += 1 #La til denne
            node = node.parent


    def uct(self, node, action, c)
        child_node = node.successors[action]
        return child_node.reward + c*sqrt( log(node.visit_count) / (1 + child_node.visit_count))

class Agent:
    ...


class MCTSAgent(Agent):
    def __init__(self, env, tree_policy, target_policy, n_episodes, n_simulations, save_interval):
        self.env = env                          # Game environment
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy)
        self.n_episodes = n_episodes            # Number of search games
        self.n_simulations = n_simulations      # Number of simulated games from leaf node
        self.save_interval = save_interval      # Number of games between network checkpoints

    def run(self):
                                                                        #   1. is = save interval for ANET (the actor network) parameters
        self.replay_buffer.clear()                                      #   2. Clear Replay Buffer (RBUF)
        self.network.initialize()                                       #   3. Randomly initialize parameters (weights and biases) of ANET 
        self.network.save()     # Save initial network

        for ep in self.n_episodes:                                      #   4. For ga in number actual games:
            env_actual = self.env.reset().copy()                        #       (a) Initialize the actual game board (Ba) to an empty board.
            root = MCNode(env_actual.get_observation())                 #       (b) sinit ← starting board state
            mct = MCTree(root, self.tree_policy, self.target_policy)    #       (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit 

            while not env_actual.is_terminal():                         #       (d) While Ba not in a final state:
                #env_sim = env_actual.copy()                             #           • Initialize Monte Carlo game board (Bmc) to same state as root. 

                for sim in self.n_simulations:                          #           • For gs in number search games:
                    env_sim = env_actual.copy()
                    leaf, search_game = mct.search(env_sim)             #               – Use tree policy Pt to search from root to a leaf (L) of MCT. Update Bmc with each move.
                    #Expanding if leaf has been visited before
                    if leaf.visit_count > 0:
                        leaf = mct.node_expansion(leaf, search_game) 
                        # Assuming first new child node chosen after expansion
                        action = search_game.get_legal_actions()[0]
                        leaf = leaf.successors[action]                    
                    reward = mct.rollout(leaf)                          #               – Use ANET to choose rollout actions from L to a final state (F). Update Bmc with each move. 
                    mct.backprop(leaf, reward)                          #               – Perform MCTS backpropagation from F to root.
                
                D = mct.get_distribution()                              #           • D = distribution of visit counts in MCT along all arcs emanating from root. 
                self.replay_buffer.add(root, D)                         #           • Add case (root, D) to RBUF
                action = np.argmax(D)                                   #           • Choose actual move (a*) based on D
                root = root.successors[action]                          #           • Perform a* on root to produce successor state s*
                mct.change_root(root)                                   #           • In MCT, retain subtree rooted at s*; discard everything else.
                env_actual.move(action)                                 #           • Update Ba to s*

            batch = self.replay_buffer.fetch_minibatch()
            self.network.train(batch)                                   #       (e) Train ANET on a random minibatch of cases from RBUF 

            if ep % self.save_interval == 0:                            #       (f) if ga modulo is == 0:
                self.network.save()                                     #           • Save ANET’s current parameters for later use in tournament play.
