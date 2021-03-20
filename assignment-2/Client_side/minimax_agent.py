
class MinimaxNode:
    '''
    Object representing one node in the search tree.
    Used in both Minimax and Minimax w/AlphaBeta-pruning.
    '''

    def __init__(self, state, action=None, agent_index=0, successors=None, utility=None, alpha=None, beta=None):
        self.state = state
        self.action = action # Holds the action taken to get to this node
        self.agent_index = agent_index
        self.successors = successors if successors else []
        self.utility = utility
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def create_minimax_node(state, action, agent_index):
        return MinimaxNode(
            state=state,
            action=action,
            agent_index=agent_index
        )
    
    @staticmethod
    def create_alphabeta_node(state, action, agent_index, alpha, beta):
        return MinimaxNode(
            state=state,
            action=action,
            agent_index=agent_index,
            alpha=alpha,
            beta=beta
        )

    def __str__(self):
        if self.alpha and self.beta:
            return "Agent {}: [utility={}, α={}, β={}]".format(
                self.agent_index,
                self.utility,
                self.alpha,
                self.beta
            )
        else:
            return f"Agent {self.agent_index}: [utility={self.utility}]"


class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()

    
class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    @staticmethod
    def minimax_multiagent(node, successor_gen, utility_func, terminal_func, depth, num_adversaries=1):
        """
        Implementation of Minimax, supporting an arbitary number of adversaries.

        Input:
            node:               Node encapsulating the state to be evaluated.
            successor_gen:      Generator which yields all successor states with corresponding 
                                action for the agent specified.
                                    (state, agent_index) -> yield (successor_state, action)
            utility_func:       State evaluation function.
                                    state -> utility
            terminal_func:      Function that determines whether the given state is a terminal state.
                                    state -> True or False
            depth:              Depth of search tree.
            num_adversaries:    Number of adversaries

        Returns:
            A MinimaxNode which represents the optimal successor node.
            Optimal action is given by the 'action' attribute of the returned node.
        """
        # Base case
        if depth == 0 or terminal_func(node.state):
            node.utility = utility_func(node.state)
            return node

        # Calculate agent index and depth for immediate successors
        next_agent = (node.agent_index + 1) % (num_adversaries + 1)
        next_depth = depth - 1 if (next_agent == 0) else depth

        # First, determine the objective at current level;
        # max if agent is PacMan, min otherwise
        _min_or_max = max if (node.agent_index == 0) else min

        # Generate successor nodes
        for state, action in successor_gen(node.state, node.agent_index):
            v = MinimaxNode.create_minimax_node(state, action, next_agent)
            node.successors.append(v)
            
            # Evaluate successor
            MinimaxAgent.minimax_multiagent(v, successor_gen, utility_func, terminal_func, next_depth, num_adversaries)

        # Find best successor, inherit utility from the selected node
        best_successor = _min_or_max(node.successors, key=lambda successor: successor.utility)
        node.utility = best_successor.utility
        return best_successor


    @staticmethod
    def successor_gen(state, agent_index):
        actions = state.getLegalActions(agent_index)
        yield from ((state.generateSuccessor(agent_index, action), action) for action in actions)
        # yield from map(lambda action: (state.generateSuccessor(agent_index, action), action), actions)
        # yield from zip(map(lambda action: state.generateSuccessor(agent_index, action), actions), actions)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        current_node = MinimaxNode(gameState)
        optimal_node = MinimaxAgent.minimax_multiagent(
            node=current_node, 
            successor_gen=self.successor_gen, 
            utility_func=lambda state: state.getScore(), 
            terminal_func=lambda state: state.isWin() or state.isLose(),
            depth=self.depth, 
            num_adversaries=gameState.getNumAgents() - 1 
        )
        return optimal_node.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    @staticmethod
    def alphabeta_multiagent(node, successor_gen, utility_func, terminal_func, depth, num_adversaries=1):
        """
        Implementation of Minimax with alpha-beta pruning, 
        supporting an arbitary number of adversaries.

        Input:
            node:               Node encapsulating the state to be evaluated.
            successor_gen:      Generator which yields all successor states with corresponding 
                                action for the agent specified.
                                    (state, agent_index) -> yield (successor_state, action)
            utility_func:       State evaluation function.
                                    state -> utility
            terminal_func:      Function that determines whether the given state is a terminal state.
                                    state -> True or False
            depth:              Depth of search tree.

            (Optional)
            num_adversaries:    Number of adversaries, defaults to 1.

        Returns:
            A MinimaxNode which represents the optimal successor node.
            Optimal action is given by the 'action' attribute of the returned node.
        """
        # Base case
        if depth == 0 or terminal_func(node.state):
            node.utility = utility_func(node.state)
            return node

        # Calculate agent index and depth for immediate successors
        next_agent = (node.agent_index + 1) % (num_adversaries + 1)
        next_depth = depth - 1 if (next_agent == 0) else depth

        # Determine the objective at current level;
        # max if agent is PacMan, min otherwise
        _min_or_max = max if (node.agent_index == 0) else min

        # Generate successor nodes
        for state, action in successor_gen(node.state, node.agent_index):
            v = MinimaxNode.create_alphabeta_node(state, action, next_agent, node.alpha, node.beta)
            node.successors.append(v)
            
            # Evaluate successor
            AlphaBetaAgent.alphabeta_multiagent(v, successor_gen, utility_func, terminal_func, next_depth, num_adversaries)
            
            # Check if current subtree can be pruned
            if _min_or_max == max:
                if v.utility > node.beta: # actually >=
                    node.utility = v.utility
                    return v
                node.alpha = max(node.alpha, v.utility)
            else:
                if v.utility < node.alpha: # actually <=
                    node.utility = v.utility
                    return v
                node.beta = min(node.beta, v.utility)

        # Find best successor, inherit utility from the selected node
        best_successor = _min_or_max(node.successors, key=lambda successor: successor.utility)
        node.utility = best_successor.utility
        return best_successor


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        current_node = MinimaxNode(gameState, alpha=-float('inf'), beta=float('inf'))
        optimal_node = AlphaBetaAgent.alphabeta_multiagent(
            node=current_node,
            successor_gen=MinimaxAgent.successor_gen, 
            utility_func=lambda state: state.getScore(),
            terminal_func=lambda state: state.isWin() or state.isLose(),
            depth=self.depth,
            num_adversaries=gameState.getNumAgents() -1 
        )
        return optimal_node.action
