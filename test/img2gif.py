import imageio

# Create a list of filenames
filenames = [f"frame_{i}.png" for i in range(1, 149)]

# Create a GIF
with imageio.get_writer('my_animation.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, remove the individual frame files if they are no longer needed
import os
for filename in filenames:
    os.remove(filename)