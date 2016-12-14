import random
from collections import OrderedDict

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    states_names = [(('light', l), ('oncoming', oa), ('left', la), ('right', ra), ('next_waypoint', na)) for l in ['green', 'red'] for oa in Environment.valid_actions for la in Environment.valid_actions for ra in Environment.valid_actions for na in Environment.valid_actions]
    
    #set the initial Q(state, action) value
    # if direction is (0,0) (at the destination): Q(s,a) = 10
    # for other Q(s, a), initial value = 0
    Q = OrderedDict()
    for sn in states_names:
        for a in Environment.valid_actions: 
            Q[(sn, a)] = 0
            #Q[(sn, a)] = random.uniform(-10,10)          
    N = 0
            
    #set parameters for Q-Learning:
    alpha = lambda t: 1.0/(t+1)
    gamma = 0.1
    update_prob = 0.1 # when choosing the action, 90% times choose the optimal action, 5% randomly choose an action
     
    
    
    
    #def __init__(self, env, g, p):
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.destination = None
        self.state = None
        self.num_neg_rewards = 0 # this keeps track the number of negative trials
        self.total_rewards = 0 # this tracks the total rewards
        self.num_circles = 0 #this tracks if the agent is taking circles over time
        # A circle is simplified as right->[None]* ->right->[None]*->right->[None]*->right or left->[None]* ->left->[None]*->left->[None]*->left
        self.num_r_l = 0 # this tracks the number of continuous right or left turns, turn right: -1, turn left: +1
        
        #set parameters for Q-Learning:
        #LearningAgent.alpha = lambda t: 1/(t+1)
        #LearningAgent.gamma = float(g)
        #LearningAgent.update_prob = float(p) # when choosing the action, 90% times choose the optimal action, 5% randomly choose an action

         

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination = destination
        self.state = None
        self.total_rewards = 0
        self.num_neg_rewards = 0
        self.num_circles = 0
        self.num_r_l = 0
        
        LearningAgent.N = LearningAgent.N + 1
        #debug: see if Q changes
        #if LearningAgent.N % 10 ==0:
         #   print ('***')
          #  for sn in LearningAgent.states_names:
           #     for a in Environment.valid_actions: 
            #        if LearningAgent.Q[(sn, a)] != 0: 
             #           print ('({}, {}, {:.2f})'.format(sn, a, LearningAgent.Q[(sn, a)]))
            #print ('***')
            
       
        


    def update(self, t):
        # Gather inputs, those are the only thing we know, we do not know the location, the where the destination is, etc. 
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)  
        
        


        # TODO: Update state        
        self.state = self.gen_state(inputs, self.next_waypoint)

        
        # TODO: Select action according to your policy        
        if random.random() < LearningAgent.update_prob:
            action=random.choice(self.env.valid_actions)
        else:
            max_action = self.find_max_Q(self.state)

            if len(max_action)==1: action = max_action[0]
            #have more than one actions which gives the max q value
            else: 
                for a in max_action:                    
                    if a == self.next_waypoint: action = a
                    else: 
                        action = random.choice(max_action)
        
   
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0: self.num_neg_rewards += 1
        self.total_rewards += reward
        
        # After choose the optimal action, get the Q(S*, A*)
        # S* is the next state, need to get the input at 
        next_state = self.gen_state(inputs = self.env.sense(self),   
                                    next_waypoint = self.planner.next_waypoint()
                                    )
        

        # TODO: Learn policy based on state, action, reward                
        max_next_q = LearningAgent.Q[(next_state, self.find_max_Q(next_state)[0])]
                
        LearningAgent.Q[(self.state, action )] = (1-LearningAgent.alpha(t)) * LearningAgent.Q[(self.state, action)] + LearningAgent.alpha(t) * (reward+LearningAgent.gamma * max_next_q)
        
        # Keep track the number of circles
        if action =='right':
            if self.num_r_l <= 0: self.num_r_l -= 1
            else: self.num_r_l = -1
        elif action == 'left':
            if self.num_r_l >= 0: self.num_r_l += 1
            else: self.num_r_l = 1
        elif action == 'forward': self.num_r_l = 0
        if abs(self.num_r_l)==4: self.num_circles += 1 

        print ("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, total_num_negative_rewards = {}, num_circles = {}, total_rewards = {}".format(deadline, inputs, action, reward, self.num_neg_rewards, self.num_circles, self.total_rewards))  # [debug]
        
        
        
    def gen_state(self, inputs, next_waypoint):
        '''Generate current state'''
        return (('light', inputs['light']), ('oncoming', inputs['oncoming']), ('left', inputs['left']), ('right', inputs['right']), ('next_waypoint', next_waypoint))

    def find_max_Q(self, state): 
        '''Find a list of actions which has the maximum Q (s,a) value at this state'''
        max_q = None
        max_action = []
        for q_key, q_value in LearningAgent.Q.items():
            if q_key[0] == self.state:   
                
                if max_q is None or max_q == q_value: 
                    max_q = q_value
                    max_action.append(q_key[1])
                elif q_value < max_q: continue
                else: 
                    max_action = []
                    max_action.append(q_key[1])
        
        return max_action


def run():
    """Run the agent for a finite number of trials."""
    import sys
    # Set up environment and agent
    
    e = Environment()  # create environment (also adds some dummy traffic)
    #a = e.create_agent(LearningAgent, g=sys.argv[1], p=sys.argv[2])  # create agent
    a = e.create_agent(LearningAgent)  # create agent
    
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    #sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
