import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, OffsetBox
import matplotlib.image as mpimg
from mountain_car_env import MountainCar
from matplotlib.artist import Artist
from matplotlib import transforms
#import imutils



class AnimateMC():
    def __init__(self, actor, env, n_steps=1000, x_range=[-1.2,0.6]):
        print("\n \n\n\n")
        print("-------------------------- ENTER ANIMATE -------------------------------------")
        self.n_steps = n_steps
        print(n_steps)
        self.mc = env
        self.mc.max_steps = self.n_steps
        self.actor = actor
        self.d = (x_range[1]-x_range[0])/100
        self.x_line = np.arange(x_range[0], x_range[1], self.d)
        self.y_line = [self.get_y(x) for x in self.x_line]
        self.padding = 0.5
        self.best_x = 0
        self.best_step = 0
        self.figure_offset = 0.1
        #self.fig, self.ax = plt.subplots()
        self.fig = plt.figure(figsize=(15,10))
        self.ax = self.fig.add_subplot(111, aspect='auto', autoscale_on=False,
                                     xlim=(x_range[0]-self.padding, x_range[1]+self.padding), ylim=(-2, 2))

        print("LIST",list(np.arange(x_range[0]-self.padding, x_range[1]+self.padding, 0.25)))
        #plt.xticks = np.arange(x_range[0]-self.padding, x_range[1]+self.padding, 0.25)
        #self.ax.set_xlim(x_range[0],x_range[1])
        #self.ax.set_ylim(-0.5, self.get_y(x_range[1]))
        #self.line.set_data(self.x_line, self.y_line)

        # Mountain line
        self.line, = self.ax.plot(0,0)




        """
        self.car_img = mpimg.imread("assignment-3/mountain_car_img.png")
        self.imagebox = OffsetImage(self.car_img, zoom=0.05)
        self.ab = AnnotationBbox(self.imagebox, (0.0,0.0), frameon=False)
        print(dir(self.ab))
        self.ax.add_artist(self.ab)
        #self.imagebox.

        """




        # Texts
        self.step_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        self.left_active = self.ax.text(0.25, 0.125, '-1', transform=self.ax.transAxes, color = 'lime')
        self.left_gray = self.ax.text(0.25, 0.125, '-1', transform=self.ax.transAxes, color='darkgray')

        self.right_active = self.ax.text(0.55, 0.125, '1', transform=self.ax.transAxes, color = 'lime')
        self.right_gray = self.ax.text(0.55, 0.125, '1', transform=self.ax.transAxes, color='darkgray')

        self.idle_active = self.ax.text(0.47, 0.125, '0', transform=self.ax.transAxes, color = 'lime')
        self.idle_gray = self.ax.text(0.47, 0.125, '0', transform=self.ax.transAxes, color='darkgray')
        
        self.car = Ellipse(xy=[0,0], width=0.1, height=0.05, angle=0)
        self.ax.add_artist(self.car)
        self.car.set_clip_box(self.ax.bbox)
        self.car.set_facecolor([0.85275809, 0.37355041, 0.32941859])
        print(f"Initial animation:l x: {self.mc.x}, v: {self.mc.v}")
        self.ani = FuncAnimation(self.fig, self.animate, frames = 1, interval = 6.25, blit=False, init_func=self.init)
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
        state = self.mc.decode_state(*self.mc.get_observation())
        action = self.actor(state, *self.mc.get_observation(), training=False)[0]
        print("-------------------------------")
        print("ACTOR suggested action:", action)
        print("-------------------------------")
        
        
        #action = self.mc.random_action()
        self.mc.apply_action(action)
        angle = self.find_angle(self.mc.x)
        self.car.angle = angle
        #print("Angle: ", self.car.angle)
        self.car.set_center(self.offset_figure(self.mc.x, angle))
        trans = transforms.Affine2D().rotate(angle).translate(*self.offset_figure(self.mc.x,angle))
        #self.ab.set_transform(trans)

        #self.ab.xybox = self.offset_figure(self.mc.x,angle)
        
        
        
        # self.ab.set_center((0.5,0.5))
        #self.ab.update_positions((0.5,0.5))
        #self.ab.xy = self.offset_figure(self.mc.x, angle)
        #self.car.set_center((self.mc.x, self.get_y(self.mc.x)))
        self.step_text.set_text('Timestep = %.1i' % self.mc.step)
        self.update_action_text()

        return 
    


    """
    def rotate_bound(self,image, angle):
    # grab the dimensions of the image and then determine the
    # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))
    """
    def offset_figure(self, x, angle):
        angle = angle/360 * 2 * math.pi + math.pi/2
        print(f"Angle: {angle}")
        print(f"x_angle_dir_ {math.cos(angle)}")
        offset = self.figure_offset
        #TESTING
        offset = 0.1
        new_x = x + offset * math.cos(angle)
        y = self.get_y(x)
        new_y = y + offset * abs(math.sin(angle))
        print(f"Offset figure: {offset * math.cos(angle)} , {offset * abs(math.sin(angle))}")
        print(f"Initial animation:l x: {self.mc.x}, v: {self.mc.v}")

        return (new_x, new_y)
    
    def init(self):
        self.line.set_data(self.x_line, self.y_line)
        #self.car.set_center((self.mc.x, self.get_y(self.mc.x)))
        return 


    def get_y(self,x):
        return math.cos(3*(x+math.pi/2))
    

    def find_angle(self, x):
        delta =  self.d
        slope = (self.get_y(x)- self.get_y(x-delta))/delta
        return math.atan(slope)/(2*math.pi)*360

    def update_action_text(self):
        action = self.mc.last_action
        

        if action == -1:
            self.left_active.set_text("<–––––")
            self.left_gray.set_text("")
            
            self.right_active.set_text("")
            self.right_gray.set_text("–––––>")

            self.idle_active.set_text("")
            self.idle_gray.set_text("0")
            return

        if action == 0:
            self.left_active.set_text("")
            self.left_gray.set_text("<–––––")
            
            self.right_active.set_text("")
            self.right_gray.set_text("–––––>")

            self.idle_active.set_text("0")
            self.idle_gray.set_text("")
            return

        if action == 1:
            self.left_active.set_text("")
            self.left_gray.set_text("<–––––")
            
            self.right_active.set_text("–––––>")
            self.right_gray.set_text("")
            
            self.idle_active.set_text("")
            self.idle_gray.set_text("0")
            return