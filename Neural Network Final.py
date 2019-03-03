"""
Vidur Singh. Machine Learning Homework. 29 Sep 2018. All rights reserved.


This program opens a training data file. It then proceeds to initilaize weights and other parameters to train the neural network.
For every example in the training data file, it finds the appropriate output using a forward pass and then proceeds to adjust all
the weights after doing a backward pass.

After the training process is done, it then opens a testing data file and goes through the entire file trying to predict the output
for each input using the trained neural network. It keeps a count of the cumalative error over the process and outputs the same.

An interesting feature of this program is that it gives you the choice to enter multiple comma seperated values for hidden layer
neurons. It then runs the entire program for each value. Finally it outputs multiple graphs using which one can compare the efffect of
the number of neurons in the hidden layer. Please attempt to use this feature. 

If you dont want to use this feauture, just enter one value when prompted.

There is a section of the code labelled #Initializing all variables in which you can change parameters of how the code runs. You can
change the upper limit of epochs run or the J threshold or the value of eta.

Have fun!
"""

"""
Interesting note on the graphs outputted by the program:

There are three graphs outputed by the program. All three graphs are representations of very interesting data. The decriptions are as follows:

1) The first graph is the output of J vs Epochs. The different coloured curves refer to a different number of neurons in the hidden layer.
   Clearly, the error reduces in the same distinctive pattern for each curve. The rate of the reduction is very high to begin with and then suddenly reduces.
   It is remarkable that the rate of reduction of error is much larger for a higher number of hidden layer neurons.

2) The second graph shows the DIFFERENCE of testing error(of last epoch) to training error. I wanted to graph this data because sometimes, the nerual
   network finds minima which are not actually the global minima. In this case, the absolute error is very high which skews the sense of the graph. However
   if the same minima is present in both the testing and the training data, then, y (= Diff(testing- training) should be low. If y is high, then
   it suugests that the underlying function is not being approximated to the same minima. If y is low, it suggests that the same function is being
   approximated. 

3) The third graph shows the absolute error vs number of neurons in hidden layer. Interesting to note that the absolute testing error does reduce with an
   increase in number of neurons. This is strange. I would assume that the average error would in fact decrease until a certain number of hidden layer
   neurons and then increase with increase in hidden layer neurons(because of over fitting). However that does not seem to be happening. I currently do not
   have an explanation for this behaviour.
"""
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
print ("Program will ask you to enter number of hidden layer neurons for each run.\n\
Please enter :3,6,12. Proram will then output error curves \nfor 3,6,12 hidden layer neurons respectively \n\
in order of blue, green, red, yellow \n\n\n")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions:
    
#Activation Function. Generates a sigmoid. Referred to in class notes as (f)
def activationfunction(x):
    ans = float(1/(1 + ((np.e)**(-x))))
    return(ans)

#returns derivative of activation function at a point  
def derivativeactivation(x):
    sigmoid = activationfunction(x)
    der= sigmoid*(1-sigmoid)
    return der

#creating small weight array (notice small "w" and "a") with appropriate number of weights in format [[w11, w12], [w21, w22],...]
def weightarraygenerate(hiddencount, inputcount):
    weightarray = []
    for neuron in range(0, hiddencount):
        weightadd = []
        weightadd = [random.uniform(-0.5, 0.5) for parameter in range(0, inputcount)]
        weightarray = weightarray + [weightadd]
    return (weightarray)

#creating capital WeightArray (notice captial "W" and "A") with appropriate number of weights in format [[W11, W12], [W21, W22],...] which in this case will just be
# [[W11, W12, W13],]
def WeightArrayGenerate(outputcount, hiddencount):  
    WeightArray = []
    for neuron in range(0, outputcount):
        WeightAdd = []
        WeightAdd = [random.uniform(-0.5, 0.5) for neuron in range(0, hiddencount)]
        WeightArray = WeightArray + [WeightAdd]
    return (WeightArray)

#creating the multiple lists involved in storing the values of S_j, h_j, S_i, yhat_i, dell_i, dell_j
def lists(hiddencount, outputcount):        
    #creating array S_j for each j (ie, each neuron in hidden layer)
    S_j= [0 for i in range(0, hiddencount)]

    #creating array h_j for each j (ie, each neuron in hidden layer)
    h_j= [0 for i in range(0, hiddencount)]

    #creating array S_i for each i (ie, each neuron in output layer)
    S_i= [0 for i in range(0, outputcount)]

    #creating array yhat_i for each i (ie, each neuron in output layer)
    yhat_i= [0 for i in range(0, outputcount)]

    #creating array dell_i for each i (ie, each neuron in output layer)
    dell_i= [0 for i in range(0, outputcount)]
    
    #creating array dell_j for each j (ie, each neuron in hidden layer)
    dell_j= [0 for i in range(0, hiddencount)]

    outputlist = [S_j, h_j, S_i, yhat_i, dell_i, dell_j]
    return outputlist

#function for forward pass. Output is h_j, yhat_i, J for every example (function is called L*epochs where L is total number of examples in training data.) 
def forwardpass(actualinput, hiddencount, outputcount, S_j, h_j, S_i, yhat_i, dell_i, dell_j, J):          
    for neuron in range(0, hiddencount):  #for each neuron
        for parameter in range(0, inputcount):    #for each parameter
            S_j[neuron] = S_j[neuron] + weightarray[neuron][parameter]*actualinput[parameter]
        h_j[neuron] = activationfunction(S_j[neuron])


    #find yhat_i:
    for outputneuron in range(0, outputcount):  #for each neuron in output layer
        for hiddenneuron in range(0, hiddencount): #for each neuron in hidden layer 
            S_i[outputneuron]= S_i[outputneuron] + h_j[hiddenneuron]*WeightArray[outputneuron][hiddenneuron]
        yhat_i[outputneuron]= activationfunction(S_i[outputneuron])
        J= J+ (((yhat_i[outputneuron])- example[1])**2)
    return [h_j, yhat_i, J]

#taking input
#opening input file. creating an array of arrays where each element is of the form [training input, desired output]
def data(filename):
    file = open(filename, "r")
    data= file.readlines()
    dataclean= []

    for i in data:
        lstadd=[]
        element= i.strip().split(" ")  #remove excess "\n" etc and split with space
        for x in element:
            lstadd = lstadd + [float(x)]  #convert from scientific notation to decimal notaion using float()
        dataclean= dataclean + [lstadd]  #append to trainingdataclean array

    #deep copy does not reflect the changes made in original list. generally useful to use deep copy in case original list gets mutated or reused.
    data= copy.deepcopy(dataclean)
    file.close()
    return(data)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Initializing all variables

inputcount= 1+1     #no of input parameters (one parameter + bias in this case)
hiddencount= 2      #no of hidden layer neurons
layercount= 1       #no of hidden layers
outputcount = 1     #no of output neurons
eta= 0.1            #eta
Jthreshold= 0.0003    #threshold for error
J= 100              #initializing value to some large value for first iteration of while loop
epochs= 2000        #total number of epochs (upper bound, can be changed)
Jevolution= []      #keeps track of how J evovles over every epoch

weightarray = weightarraygenerate(hiddencount, inputcount)   #initializing weight array
WeightArray = WeightArrayGenerate(outputcount, hiddencount)  #initializing Weight Array
S_j, h_j, S_i, yhat_i, dell_i, dell_j= lists(hiddencount, outputcount)  #initializing assorted lists
hiddencounts =  [int(i) for i in input("number of hidden neurons for each training run (for example, enter:3,6,12)").split(",")]  #hidden counts for which to run epochs. 
plotJs=[]  #every item in this list will represent one curve on the error graph
testingerrors=[]
trainingerrors=[]
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#forward pass for each hiddencount in hiddencounts (to make graph of error curves using different number of hidden neurons in hidden neuron layer)

for hiddencount in hiddencounts:
    #reset appropriate variables. 
    Jevolution= []
    weightarray = weightarraygenerate(hiddencount, inputcount)
    WeightArray = WeightArrayGenerate(outputcount, hiddencount)
    S_j, h_j, S_i, yhat_i, dell_i, dell_j= lists(hiddencount, outputcount)

    #doing forward pass
    trainingdata= data("TrainingData.txt")  #getting training data using data function
   
    currentepoch =0  #count for epoch
    
    while J>Jthreshold and currentepoch<epochs:   #keep running epochs until you reach upper bound on epochs or threshold for J
        currentepoch+=1
        Jevolution= Jevolution + [J]   #add current J to J evolution list. Will later use this list to create the graph for error. 
        J = 0 #reset J for this epoch
        #for each example:
        # 1) Find h_j: activation function(sum the parameter*weight for each neuron for each parameter(2 parameters).)
        # 2) Find yhat_i
        for example in trainingdata:  #for each example
            S_j, h_j, S_i, yhat_i, dell_i, dell_j= lists(hiddencount, outputcount)  #reset this data 

            #training data comprises of [[input, output], [input, output]...]. To find S_j, I want [input parameter, input parameter...].
            #let us call this actualinput. Since there is only one input parameter and one bias parameter, I construct the actual input as follows:
            actualinput = [example[0],1]  #[input, bias]

            #find h_j, yhat_i, J
            h_j, yhat_i, J= forwardpass(actualinput, hiddencount, outputcount, S_j, h_j, S_i, yhat_i, dell_i, dell_j, J)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------        #Backward Pass
#backward pass
            
            #Compute dell_i (for each output neuron)
            for neuron in range(0, outputcount):       
                dell_i[neuron]= (example[1]-yhat_i[neuron])*derivativeactivation(S_i[neuron])

            #computing dell_j
            for hiddenneuron in range(0, hiddencount):
                for outputneuron in range(0, outputcount):
                    dell_j[hiddenneuron]= dell_j[hiddenneuron] + dell_i[outputneuron]*WeightArray[outputneuron][hiddenneuron]*derivativeactivation(S_j[hiddenneuron])

            #update Capital Weights
            for outputneuron in range(0, outputcount):
                for hiddenneuron in range(0, hiddencount):
                    WeightArray[outputneuron][hiddenneuron]= WeightArray[outputneuron][hiddenneuron] + eta*(dell_i[outputneuron])*h_j[hiddenneuron]

            #update small weights
            for hiddenneuron in range(0, hiddencount):
                for inpt in range(0, inputcount):
                    weightarray[hiddenneuron][inpt]= weightarray[hiddenneuron][inpt] + eta*dell_j[hiddenneuron]*actualinput[inpt]
    plotJs  += [Jevolution]  #every item in plotJ refers to one curve in the error graph. After all epochs are done, add the curve to plotJs
    trainingerrors+=[J]
    print("Neural Network is now trained. Ran", currentepoch, "epochs. J for last epoch was", J, ". Number of neurons:", hiddencount)
    #The neural network is now "trained". ie, appropriate weights have been assigned for all combinations. 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------    #Testing
    #Testing.

    #running a forward pass on the entire testing data and calculating J
    #opening input file. creating an array of arrays where each element is of the form [testing input, desired output]
    testingdata= data("TestingData.txt")
    J = 0

    #for each example:
    # 1) Find h_j: activation function(sum the parameter*weight for each neuron for each parameter(2 parameters).)
    # 2) Find yhat_i
    for example in testingdata:  #for each example
        S_j, h_j, S_i, yhat_i, dell_i, dell_j= lists(hiddencount, outputcount)
        #testing data comprises of [[input, output], [input, output]...]. TO find S_j, I want [input parameter, input parameter...].
        #let us call this actualinput. Since there is only one input parameter and one bias parameter, I construct the actual input as follows:
        actualinput = [example[0],1]  #input, bias
        #find h_j:
        h_j, yhat_i, J= forwardpass(actualinput, hiddencount, outputcount, S_j, h_j, S_i, yhat_i, dell_i, dell_j, J)

    testingerrors+=[J]
    print("Using testing data, loss was", J)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------  
#Plotting graph of the training.

plottingerrors = [0 for i in range(0, len(testingerrors))]
for i in range(0, len(plottingerrors)):
    plottingerrors[i]= (testingerrors[i]-trainingerrors[i])
fig, ax = plt.subplots()
ax.set_color_cycle(['blue', 'green', 'red', 'yellow'])
for dataset in plotJs:
    plt.plot(dataset[1:])
plt.ylabel('Error(J)')
plt.xlabel('Epochs')
plt.show()
plt.ylabel('Testing error- Training error')
plt.xlabel('Neurons in hidden layer')
plt.text(1,0,'Large distance from line means minima found during training is not present during testing')
plt.plot([i for i in range(1, hiddencount+1)],[0 for i in range(hiddencount)])
plt.plot(hiddencounts,plottingerrors, "ro")
plt.show()
plt.ylabel('Absolute testing error')
plt.xlabel('Neurons in hidden layer')
plt.plot(hiddencounts,testingerrors, "ro")
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
