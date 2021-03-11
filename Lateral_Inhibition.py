 
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from skimage import io, color, exposure, img_as_ubyte, img_as_float
from scipy.signal import convolve2d as scipy_conv2d
# Question 1

# Computes activation levels for a given list
def compute_activation_levels(lis: List[float]):
    w = 0.1
    A = np.array(lis)
    # Here we chose to ignore to the edge cells.
    A = A[1:-1]-w*(A[0:-2]+A[2:]) 
    
    return A

# Compues a list of thresholded activations for a given list and threshold
def compute_activations(activation_list : List[float], threshold: float) -> List[int]:
    
    active = np.array(activation_list)  
    active[active < threshold] = 0
    active[active >= threshold] = 1

    return list(active)
    


# Input for Q1
I = [1,1,1,1,1,0,0,0,0,0]

# Defines the activation levels for input I (Q1)
activation_levels = compute_activation_levels(I)



# Computes the actionvations on I with threshold
    # When the threshold is set to 1-w then we get edge detection (on edge with 0)
    # So in our case 1-0.1 = 0.9
activations = compute_activations(activation_levels, 0.9)
max_index = np.argmax(activation_levels)

# Prints results
print(f"Activation levels for Q1: {activation_levels}\nNeurons Activated: {activations}\n1 means active, 0 means inactive.")
print(f"The index of the cell(s) with the highest activation level is {max_index} with a value of {activation_levels[max_index]}\n")


# Question 2

# Input for Q2
I2 = [0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,3 ,3 ,3 ,4 ,4 ,4 ,5 ,5 ,5] 

# Calculates and prints activation levels
activation_levels2 = compute_activation_levels(I2)
print(f"Activation levels for Q2: {activation_levels2}")

# Removes edges of I2
I2_no_edge = np.array(I2[1:len(I2)-1])

# Min-max standardization on input and activation levels
I2_no_edge_norm = (I2_no_edge-min(I2_no_edge))/(max(I2_no_edge)-min(I2_no_edge))
activation_levels2_norm = (activation_levels2-min(activation_levels2))/(max(activation_levels2)-min(activation_levels2))



# Visualize the activation levels in Q2.
# Note a3 is used in Question 3.
fig, (a1,a2,a3) = plt.subplots(3,1)
a1.imshow([I2_no_edge_norm], cmap='gray')
a1.set_title("Q2 Input")
a1.set_xticks([i for i in range(1,17)])

a2.imshow([activation_levels2_norm], cmap='gray')
a2.set_title("Q2 Activation levels")
a2.set_xticks([i for i in range(1,17)])

# Question 3


kernel = [-0.1,1,-0.1]
resQ1, resQ2 = np.convolve(I, kernel, "valid"), np.convolve(I2, kernel, "valid")
print(f"Activation levels for Q3: I in Question 1{resQ1},\n I in Question 2 {resQ2}")
a3.imshow([resQ2], cmap='gray')
a3.set_title("Q3 Activation levels for I in Q2")
a3.set_xticks([i for i in range(0,17)])

# Question 4 

from scipy.signal import convolve2d as scipy_conv2d

img = io.imread("resources/MonaLisa.jpg")
img_float = img_as_float(img)


# Generates a kernel as specified in the assignment
def kernel(w, scale=3, center_val=1):
    center = np.ones((scale,scale)) *center_val
    complete = np.pad(center, scale ,"constant",constant_values=w)
    return complete


# This function computes a 2d convolution
def convolve2d(f_in,w,**kwargs):
    w = np.rot90(w,2)
    return scipy_conv2d(f_in,w,**kwargs)

fig = plt.figure(figsize= (10,10))
plt.title('Grayscaled Mona Lisa')
plt.imshow(img_float, 'gray')

ker = kernel(-0.1, scale=3, center_val=1)

# Uncomment this to just get a single convoled image of Mona Lisa
#n = 2
#img_kern = convolve2d(img_float, kernel(-0.1, n, center_val=1))
#fig = plt.figure(figsize= (10,10))
#plt.imshow(img_kern, 'gray')
#plt.title(f'Convolution with n = {n}')



# Setup for testing different levels of n.
r, c = 5, 3
fig, axes = plt.subplots(r, c)
fig.set_figwidth(15)
fig.set_figheight(15)
fig.tight_layout()

# This will create a picture with the Mona Lisa convolved at different levels of n.
p=1
for i in range(r):
    for j in range(c): 
        img_conv = convolve2d(img_float, kernel(-0.1, p, center_val=1))
        axes[i, j].imshow(img_conv, 'gray')
        axes[i, j].set_title(f"n = {p}")
        p += 1


fig2, axes2 = plt.subplots(r, c)
fig2.set_figwidth(15)
fig2.set_figheight(15)
fig2.tight_layout()

# This will create a picture with the Mona Lisa convolved at different levels of n with a threshold function applied.
p=1
for i in range(r):
    for j in range(c): 
        img_conv = compute_activations(convolve2d(img_float, kernel(-0.1, p, center_val=1)), 0.9)
        axes2[i, j].imshow(img_conv, 'gray')
        axes2[i, j].set_title(f"n = {p}")
        p += 1

fig3, axes3 = plt.subplots(r, c)
fig3.set_figwidth(15)
fig3.set_figheight(15)
fig3.tight_layout()

# Question 5

img = io.imread("resources/hermann.jpg")
img_float = img_as_float(img)

# This will create a picture with the Hermann Grid convolved at different levels of n.
p=1
for i in range(r):
    for j in range(c): 
        img_conv = convolve2d(img_float, kernel(-0.1, p, center_val=1))
        axes3[i, j].imshow(img_conv, 'gray')
        axes3[i, j].set_title(f"n = {p}")
        p += 1



plt.show()  

exit(0)
