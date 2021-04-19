from mountain_car_env import MountainCar
from animate import AnimateMC
import random


def run_simulation(n_timesteps):
    #mc = MountainCar([-1.2,0.6], [-0.07, 0.07], n_timesteps)
    #ani = AnimateMC()
    #print("Initial random x: ", mc.x)
    """


    for step in range(n_timesteps):
        # GET ACTION FROM NETWORK
        action = mc.random_action()
        print("Action: ", action)
        mc.apply_action(action)
        mc.visualize()
    """
def main():
    run_simulation(1000)


main()
