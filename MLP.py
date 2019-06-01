import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#constant bias input
biasInput=1
#eta
learningrate=0.1
#momentum
alpha=0.9
#number of hidden units in hidden layer
hidden_units=100
#total number of epochs
epochs=50

#input layer weights: wij
input_weight = np.random.uniform(-0.05,0.05,(784,hidden_units))
#output layer weights: wjk
hidden_weight = np.random.uniform(-0.05,0.05,(hidden_units,10))

#input layer bias weights applied to 2/50/100 neurons in hidden layer
inputlayer_bias_weights = np.random.uniform(-0.05,0.05,(1,hidden_units))
#hidden layer bias weights applied to 10 neurons in output layer
hiddenlayer_bias_weights = np.random.uniform(-0.05,0.05,(1,10))

#previous delta input bias weight
delta_input_bias_weight = np.zeros((1,hidden_units))
#previous delta hidden bias weight
delta_hidden_bias_weight = np.zeros((1,10))

#previous delta input  weight
delta_input_weight = np.zeros((784,hidden_units))
#previous delta hidden weight
delta_hidden_weight = np.zeros((hidden_units,10))

#lists to store the train and test data accuracy of each epoch
accuracy_train=[]
accuracy_test=[]

#function to permute data, preprocess data and separate data and label
def preprocess_permute_data(input_data):
    data=np.random.permutation(input_data) #permute data
    data=input_data[:,1:] #separate data
    data=data/255 #preproces data
    label=input_data[:,0] #separate label
    return data,label

#function to calculate sigma(wx)=1/(1+e^-wx)
def sigmoid_function(wx):
    return expit(wx)

#initiate forward propagation and call in backpropagation when epoch is not 0 and data is Training data
def forwardpropagation(data,label,data_name,epoch):
    global input_weight,hidden_weight,biasInput,inputlayer_bias_weights,hiddenlayer_bias_weights,delta_input_bias_weight,delta_input_weight,delta_hidden_bias_weight,delta_hidden_weight
    predict_data=[]
    label_cfm=[]
    correct=0 #counter to keep track of correct prediction in the dataset for a epoch
    for i in range(data.shape[0]): # iterate through the entire dataset
        inputData = np.reshape(data[i],(1,data[i].shape[0]))
        wx_input=np.dot(inputData,input_weight) + (biasInput*inputlayer_bias_weights)  #calculate summation of wxij
        sigma_hidden = sigmoid_function(wx_input) #calculate hj=sigma(wxij)
        wx_hidden = np.dot(sigma_hidden,hidden_weight) + (biasInput*hiddenlayer_bias_weights) #calculate summation of wjk
        sigma_output = sigmoid_function(wx_hidden) #calculate ok=sigma(wjk)
        prediction = np.argmax(sigma_output)  #get the index of the neurorn with the hightest sigma(output)
        if(epoch==49):  #if epoch is 49 then store the prediction of each dataset and label corresponding to it
            predict_data.append(prediction)
            label_cfm.append(label[i])
        if(prediction==label[i]):  #if prediction matches the label increment the correct prediction counter
            correct=correct+1
        if(data_name == 'Training' and epoch != 0): # Initiate backpropagation if it is Training dataset and epoch is not 0
            delta_input_bias_weight,delta_input_weight,delta_hidden_bias_weight,delta_hidden_weight = backpropagation(inputData,sigma_hidden,sigma_output,label[i])
    if(epoch==49): # Create confusion matrix for the 50th epoch
        cfm=confusion_matrix(label_cfm,predict_data)
        print(data_name," Confusion Matrix ")
        print(cfm)
    return (correct/len(data))*100 #calculate the percentage of accuracy of one epoch of the entire dataset

#Calculate the hidden and output error and initiate weight update
def backpropagation(data,sigma_hidden,sigma_output,label):
    global biasInput,hidden_weight,hiddenlayer_bias_weights,learningrate,alpha,delta_hidden_bias_weight,delta_hidden_weight
    target = np.insert((np.zeros((1,9))+0.1),label.astype(int),0.9)
    output_error = sigma_output * (1 - sigma_output) * (target - sigma_output) #calculate output_error=ok*(1-ok)*(t-ok)
    hidden_error = sigma_hidden * (1 - sigma_hidden) * np.dot(output_error,np.transpose(hidden_weight)) #calculate hidden_error=hj*(1-hj)*(output_error*wjk)
    delta_output = weightupdate_hiddenlayer(sigma_hidden,output_error) #update output layer weight
    delta_hidden = weightupdate_inputlayer(data,hidden_error) #update hidden layer weight
    return delta_hidden[0],delta_hidden[1],delta_output[0],delta_output[1]

#calculate wkj=wkj+n*output_error*hj
def weightupdate_hiddenlayer(sigma_hidden,output_error):
    global biasInput,hiddenlayer_bias_weights,hidden_weight,learningrate,alpha,delta_hidden_weight,delta_hidden_bias_weight #calculating n*output_error*hj of bias weight
    delta_bias = (learningrate * np.dot(output_error,biasInput)) + (alpha * delta_hidden_bias_weight)  #wij=wij+n*output_error*hj of bias weight
    hiddenlayer_bias_weights = hiddenlayer_bias_weights + delta_bias #calculate n*output_error*hj of input weight
    delta_weight = (learningrate * np.dot(np.transpose(sigma_hidden),output_error)) + (alpha * delta_hidden_weight)
    hidden_weight = hidden_weight + delta_weight #wij=wij+n*output_error*hj of hidden weight
    return delta_bias,delta_weight

#calculate wij=wij+n*hidden_error*xi
def weightupdate_inputlayer(data,hidden_error):
    global biasInput,inputlayer_bias_weights,input_weight,learningrate,alpha,delta_input_bias_weight,delta_input_weight
    delta_bias = (learningrate * np.dot(hidden_error,biasInput)) + (alpha * delta_input_bias_weight)  #calculate n*hidden_error*xi of bias weight
    inputlayer_bias_weights = inputlayer_bias_weights + delta_bias #calculate wij=wij+n*hidden_error*xi of bias weight
    delta_weight = (learningrate * np.dot(np.transpose(data),hidden_error)) + (alpha * delta_input_weight) #calculate n*hidden_error*xi of input layer weight
    input_weight = input_weight + delta_weight #calculate wij=wij+n*hidden_error*xi of input layer weight
    return delta_bias,delta_weight

#main function
def main():
    train=np.genfromtxt('C:/Users/vinay/Documents/CS545_ML/PA1/mnist_train.csv',delimiter=",") #get training data from csv file
    test=np.genfromtxt('C:/Users/vinay/Documents/CS545_ML/PA1/mnist_test.csv',delimiter=",") #get test data from csv file
    for eachepoch in range(epochs): #iterate over 50 epochs
        train_data, train_label = preprocess_permute_data(train) #permute data, preprocess data, separate training data and label
        test_data, test_label = preprocess_permute_data(test) #permute data, preprocess data, separate test data and label
        train_acc=forwardpropagation(train_data,train_label,'Training',eachepoch) #initiate forward propagation of Training data
        accuracy_train.append(train_acc) #add accuracy to list
        test_acc=forwardpropagation(test_data,test_label,'Testing',eachepoch) #initiate forward propagation of test data
        accuracy_test.append(test_acc) #add accuracy to list
    print("Accuracy of Train data : ",accuracy_train)
    print("Accuracy of Test data : ",accuracy_test)
    #generate plot of epoch vs accuarcy
    plt.plot(accuracy_train,color='green',label='Train Data Accuracy')
    plt.plot(accuracy_test,color='red',label='Test Data Accuracy')
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    image= "HiddenUnits_100.png"
    plt.title("Hidden Units 100")
    plt.savefig(image)
    plt.show()


if __name__==main():
    main()




