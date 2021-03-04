 
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from skimage import io, color, exposure, img_as_ubyte, img_as_float
from scipy.signal import convolve2d as scipy_conv2d
# Q1

def compute_activation_levels(lis: List[float]):
    w = 0.1
    A = np.array(lis)
    # Here we chose to ignore to the edge cells.
    A = A[1:-1]-w*(A[0:-2]+A[2:]) 
    
    return A



def compute_activations(activation_list : List[float], threshold: float) -> List[int]:
    
    active = np.array(activation_list)  
    active[active < threshold] = 0
    active[active >= threshold] = 1

    return list(active)
    
    



lis = [1,1,1,1,1,0,0,0,0,0]
activation_levels = compute_activation_levels(lis)

# When the threshold is set to 1-w then we get edge detection
# So in our case 1-0.1 = 0.9
activations = compute_activations(activation_levels, 0.1)
max_index = np.argmax(activation_levels)
print(f"Activation levels for Q1: {activation_levels}\nNeurons Activated: {activations}\n1 means active, 0 means inactive.")
print(f"The index of the cell(s) with the highest activation level is {max_index} with a value of {activation_levels[max_index]}\n")


# Q2

"""I2 = [0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,3 ,3 ,3 ,4 ,4 ,4 ,5 ,5 ,5] 

activation_levels2 = compute_activation_levels(I2)
print(f"Activation levels for Q2: {activation_levels2}")


inputs = np.array(I2[1:-1])
inputs_norm = (inputs-min(inputs))/(max(inputs)-min(inputs))
activation_levels2 = (activation_levels2-min(activation_levels2))/(max(activation_levels2)-min(activation_levels2))


print(activation_levels2)

plt.imshow([inputs_norm[1:-1],activation_levels2], cmap='gray')
plt.show()"""

I2 = [0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,3 ,3 ,3 ,4 ,4 ,4 ,5 ,5 ,5] 

activation_levels2 = compute_activation_levels(I2)
print(f"Activation levels for Q2: {activation_levels2}")


inputs = np.array(I2[1:-1])
print(inputs)
inputs_norm = (inputs-min(inputs))/(max(inputs)-min(inputs))
activation_levels2 = (activation_levels2-min(activation_levels2))/(max(activation_levels2)-min(activation_levels2))



# print(activation_levels2)

fig, (a1,a2,a3) = plt.subplots(3,1)
a1.imshow([inputs_norm[1:-1]], cmap='gray')
a1.set_title("Q2 Input")
a2.imshow([activation_levels2], cmap='gray')
a2.set_title("Q2 Activation levels")

# 3


# !We need to compute for both instances of the input, 
# but we will add that ourselves.¡

I2 = [0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,3 ,3 ,3 ,4 ,4 ,4 ,5 ,5 ,5] 
kernel = [-0.1,1,-0.1]
# The valid option makes sure we only calculate the convolution, 
# where we can take a valid dot product/where the number of elements in the array
# match the number of elements in the kernel.
res = np.convolve(I2, kernel, "valid")
print(f"Activation levels for Q3: {res}")
a3.imshow([res], cmap='gray')
a3.set_title("Q3 Activation levels")
#plt.show()


# 4 

from scipy.signal import convolve2d as scipy_conv2d

img = io.imread("resources/MonaLisa.jpg")
img_float = img_as_float(img)

def kernel(w, scale = 3, center_val=1):
    kernel = np.full(shape=(scale, scale), fill_value=w)
    center = scale // 2
    if scale % 2 == 0:
        r = (center-1,center+1) 
        kernel[r[0]:r[1], r[0]:r[1]] = center_val
    else:
        kernel[center,center] = center_val
    return kernel

def convolve2d(f_in,w,**kwargs):
    w = np.rot90(w,2)
    return scipy_conv2d(f_in,w,**kwargs)


img_kern = convolve2d(img_float, kernel(-0.1, 6, center_val=1))
fig = plt.figure(figsize= (10,10))
plt.imshow(img_kern, 'gray')



# Try different levels
fig, axes = plt.subplots(3, 3)

for i in range(4,12):
    img_kern = convolve2d(img_float, kernel(-0.1, i, center_val=i/10))
    axes[i//3,i%3].imshow(img_kern, 'gray')

plt.show()  