# Self-Driving Smart Cab

SmartCabs are a kind of self-driving cab which are able to transfer people from one location to another efficiently, safely and reliably. In this project, reinforcement learning techniques have been applied to train a safe and reliable SmartCab agent. There are two main goals for the SmartCab agent to achieve: learn the traffic rules quickly and get to the destination in a short time. The idea of the project and the starter codes come from [Udacity](https://www.udacity.com/). 

## Language and libraries
* Python 3.5 
* NumPy
* pandas
* matplotlib
* PyGame

## Programs
1. environment.py  
   This program contains the TrafficLight Class, Environment Class, Agent Class and DummyAgent Class.   
2. planner.py  
   This program provides a naive route planner.
3. simulator.py
   This program simulates the entire environment where the agents operate and displays a GUI by PyGame. 
4. agent.py
   This program defines a class for self-driving SmartCab by implementing the Q-Learning algorithm. 

## Reinforcement Learning
[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) is an important area of machine learning and [Q-learning](https://en.wikipedia.org/wiki/Q-learning) is a model-free reinforcement learning technique.   
* Attributes for Q-Learning:
  * States: (light status, heading direction of the upcoming vehicle, heading direction of the vehicle on the left, heading direction of the vehicle on the right, suggested heading direction of next step)
  * Actions: stay, turn left, turn right, go forwards
  * Rewards: positive for making a good action; negative for violating the traffic rules or causing a traffic accident  
       
 * Parameters for Q-Learning:
   * Discount Factor (*gamma*): 0.05
   * Learning Rate (*epsilon*): 0.05
   * Initial Quantity (*Q(S, A)*): 0
             
## Demo
Please click [here](https://youtu.be/5b8snX3oddU) to see the demo. The red car is the SmartCab. 

## Limitations
1. The environment is ideal, grid-like 
2. The route taken by SamrtCab is not always the shortest
