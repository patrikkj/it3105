import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import matplotlib.image as mpimg



image = mpimg.imread("assignment-3/mountain_car_img.png")
fig, ax = plt.subplots()
im = ax.imshow(image)
print(dir(image))
patch = patches.Rectangle()
patch = patches.Circle((260, 200), radius=200, transform=ax.transData)
#im.set_clip_path(patch)


ax.axis('off')
plt.show()