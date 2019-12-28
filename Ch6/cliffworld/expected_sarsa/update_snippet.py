class ExpectedSarsaAgent(agent.BaseAgent):
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_info["seed"])
        
        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action, n_ties = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
                # my notes
                state := S_{t+1}; self.prev_state := S_t
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action, n_ties = self.argmax(current_q)

        # Perform an update (~5 lines)
        ### START CODE HERE ###
        def compute_eps_greedy_model(current_q):
            max_action = np.argmax(current_q)  # should this be max action or selected action?
            num_greedy = sum([current_q[a] == current_q[max_action] for a in range(self.num_actions)])
            num_non_greedy = self.num_actions - num_greedy
            non_greedy_prob = self.epsilon / self.num_actions
            greedy_prob  = (1 - (num_non_greedy * non_greedy_prob)) / num_greedy
            expectation = 0
            for a in range(self.num_actions):
                prob = non_greedy_prob if current_q[a] != current_q[max_action] else greedy_prob
                expectation += prob * current_q[a]
            return expectation
            
        cond_expect = compute_eps_greedy_model(current_q)  # E_{A'} [Q(S_{t+1}, A') | S_{t+1}]
        target = reward + (self.discount * cond_expect) - self.q[self.prev_state, self.prev_action]
        self.q[self.prev_state, self.prev_action] += self.step_size * target
        ### END CODE HERE ###
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state, self.prev_action] += self.step_size*(reward - self.q[self.prev_state, self.prev_action])
        ### END CODE HERE ###
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
            n_ties (int): num ties
        """
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        
        return self.rand_generator.choice(ties), len(ties)
