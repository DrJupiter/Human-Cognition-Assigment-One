
import numpy as np
from typing import List
import matplotlib.pyplot as plt

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
activations = compute_activations(activation_levels, 0.2)
max_index = np.argmax(activation_levels)
print(f"Activation levels for Q1: {activation_levels}\nNeurons Activated: {activations}\n1 means active, 0 means inactive.")
print(f"The index of the cell(s) with the highest activation level is {max_index} with a value of {activation_levels[max_index]}\n")


# Q2

I2 = [0 ,0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,3 ,3 ,3 ,4 ,4 ,4 ,5 ,5 ,5] 

activation_levels2 = compute_activation_levels(I2)
print(f"Activation levels for Q2: {activation_levels2}")


inputs = np.array(I2[1:-1])
inputs_norm = (inputs-min(inputs))/(max(inputs)-min(inputs))
activation_levels2 = (activation_levels2-min(activation_levels2))/(max(activation_levels2)-min(activation_levels2))


print(activation_levels2)

plt.imshow([inputs_norm[1:-1],activation_levels2], cmap='gray')
plt.show()