import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, OffsetBox
import matplotlib.image as mpimg
from mountain_car_env import MountainCar
from matplotlib.artist import Artist
from matplotlib import transforms
import datetime

#import imutils
class AnimateMC():
    def __init__(self, actor, env, n_steps=1000, x_range=[-1.2,0.6]):
        print("\n \n\n\n")
        print("-------------------------- ENTER ANIMATE -------------------------------------")
        self.n_steps = n_steps
        self.mc = env
        self.mc.max_steps = self.n_steps
        self.actor = actor
        self.padding = 0.5
        self.line_padding = 0.05
        self.d = (x_range[1]-x_range[0])/100
        self.x_line = np.arange(x_range[0], x_range[1]+self.line_padding, self.d)
        self.y_line = [self.get_y(x) for x in self.x_line]
        self.best_x = 0
        self.best_step = 0
        self.figure_offset = 0.1
        self.fig = plt.figure(figsize=(15,10))
        self.ax = self.fig.add_subplot(111, aspect='auto', autoscale_on=False,
                                     xlim=(x_range[0]-self.padding, x_range[1]+self.padding), ylim=(-1.2, 1.2))
        self.fig.patch.set_alpha(0.)        # Mountain cos line 
        self.line, = self.ax.plot(0,0)
        
        # Setting image of car
        self.car_img = mpimg.imread("assignment-3/mountain_car_img.png")
        self.im = self.ax.imshow(self.car_img,interpolation='none',origin='upper',extent=[-0.0625, 0.0625, -0.025, 0.025], clip_on=True)
        
        # Fix aspect ratio and visuals of plot
        self.ax.set(adjustable="datalim")
        self.ax.set(aspect = "auto")
        self.ax.grid(False)
        plt.axis("off")
        self.ax.set_frame_on(False)
  
        # Setting up texts and arrows
        self.step_text = self.ax.text(0.15, 0.95, '', transform=self.ax.transAxes)      
        
        self.right_arrow = plt.arrow(-0.3,1.1,0.1,0.0, width = 0.0175, color="gray", head_length=0.05)
        self.left_arrow = plt.arrow(-0.6,1.1,-0.1,0.0, width = 0.0175, color="gray", head_length=0.05)
        self.idle= self.ax.text(0.441, 0.95, '0', transform=self.ax.transAxes, color = 'gray',fontsize=15, fontweight="bold")

        
        
        
        print(f"Initial animation:l x: {self.mc.x}, v: {self.mc.v}")
        
        # Create animation
        animation = FuncAnimation(self.fig, self.animate, frames = 1000, interval = 6.25, blit=False, init_func=self.init)
        
        #Setting up writing object to save animation to .mp4 file
        Writer = writers["ffmpeg"]
        writer = Writer(fps=30, metadata = {'artist':'Me'}, bitrate = 30_000)
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_base = "Mountain_car_animation"
        filename =  f"{filename_base}_{date}.mp4"
        filename2 = f"circle_ani_{date}.mov"
        animation.save(filename2, codec="png",
         dpi=100, bitrate=-1, fps=30,
         savefig_kwargs={'transparent': True, 'facecolor': 'none'})
        #animation.save(filename, writer)

        plt.show()

    # Function that will be called to update each frame in animation
    def animate(self, i):
        print(f"ANIMATION step: {self.mc.step}, x: {self.mc.x}, v: {self.mc.v}")
        
        if self.mc.is_finished():
            print("Entered finished")
            self.complete_text = self.ax.text(0.02, 0.9, 'Finished!', transform=self.ax.transAxes, color='Green')
            return
        if self.mc.is_timeout():
            print("Entered timeout")
            self.timeout_text = self.ax.text(0.02, 0.95, 'Timestep = %.1i' % self.mc.step, transform=self.ax.transAxes, color='red')
            self.step_text.set_text("")
            return 

        # ADD action (get from actor)
        state, indices = self.mc.decode_state(*self.mc.get_observation())
        action = self.actor(state, *self.mc.get_observation(), training=False)[0]
        self.mc.apply_action(action)
        angle = self.find_angle(self.mc.x)
        self.update_image(transforms.Affine2D().rotate(angle/360*2*math.pi).translate(*self.offset_figure(self.mc.x,angle)))
        self.step_text.set_text('Timestep = %.1i' % self.mc.step)
        self.update_action_text()
        return 

    # Offsets the car so that it looks like it drives on the line
    def offset_figure(self, x, angle):
        angle = angle/360 * 2 * math.pi + math.pi/2
        print(f"Angle: {angle}")
        print(f"x_angle_dir_ {math.cos(angle)}")
        offset = self.figure_offset
        #TESTING
        offset = 0.03
        new_x = x + offset * math.cos(angle)
        y = self.get_y(x)
        new_y = y + offset * abs(math.sin(angle))
        print(f"Offset figure: {offset * math.cos(angle)} , {offset * abs(math.sin(angle))}")
        print(f"Initial animation:l x: {self.mc.x}, v: {self.mc.v}")
        return (new_x, new_y)
    
    # Init function for animation
    def init(self):
        self.line.set_data(self.x_line, self.y_line)
        return 

    # Returns y coordinate based on x coordinate
    def get_y(self,x):
        return math.cos(3*(x+math.pi/2))

    # Updates the transformation data of the image
    def update_image(self, transform):

        trans_data = transform + self.ax.transData
        self.im.set_transform(trans_data)

    # Finds the angle of the slope, used for rotating the car correctly
    def find_angle(self, x):
        delta =  self.d
        #return 30
        slope = (self.get_y(x)- self.get_y(x-delta))/delta
        return math.atan(slope)/(2*math.pi)*360

    # Update the information about what action is performed
    def update_action_text(self):
        action = self.mc.last_action
        
        if action == -1:
            self.left_arrow.set(color="green")
            self.right_arrow.set(color="gray")
            self.idle.set(color="gray")
            return
        if action == 0:
            self.left_arrow.set(color="gray")
            self.right_arrow.set(color="gray")
            self.idle.set(color="green")
            return
        if action == 1:
            self.left_arrow.set(color="gray")
            self.right_arrow.set(color="green")
            self.idle.set(color="gray")
            return