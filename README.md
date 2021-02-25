# Human-Cognition-Assigment-One

## Q1 - Lateral Inhibition

Input: A list of N-length

Compute: Activation of each cell based on the input of each neighbouring cell.
    A_i = I_i - w*(I_(i-1) + I_(i+1))

> Alternatively this can be expressed as
> > A_i = Sum_(j=-1)^(1) I(i+j)*F(j)
> > F(j) = -w if j != 0

Specifics:
The weight w is set to `w = 0.1` and `I = [1, 1, 1, 1, 1, 0, 0 ,0 ,0 ,0]`

Optional parameter specifying the threshold `t`, for each cell, the activation is set to 0 if the activation is smaller than the threshold.

The threshold function is something which we have to set and adjust and observe the results of.

### Subquestions:

- What are the activation levels of the cells?
- Which cells have the highest activation level?
- Neurons are only active when the sum of the activation they receive from
  other neurons exceed a certain threshold. Equip the cells with a threshold
  function so that the cells function as an edge detector. Describe how your
  threshold function works and its output.


## Q2 - Optical Illusion

The same as Q1 except for 

`I = [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5]`
`-> N = 18`
(No treshold)

### Sub Questions:

- What is the activation levels of the cells?(No threshold)
- How does the activation profile fit the input profile? Describe the difference. 
- What optical illusion does this describe?





> The illusion could be the gradienty illusion with the gray colors
> ("Mach  Bands")



## Q3 - Convolution



## Q4 - Mona Lisa



## Q5 - Hermann Grid

Hello ppl of this group,
today we will eat potato soup and.
Enjoy

I love potatoes

I like ____

