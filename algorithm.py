#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    Data is imported and stored in a 2-d list, where each row is a different
    instance, col[0] = x1, col[1] = x2, and col[2] = c(x).
    Learning rate is set to 0.5.
    Weights from [0,1] are calculated using random(). Then, using the training 
    set and the cooresponding weights, h(x) is calculated, compared to c(x),
    and if h(x) != c(x), new weights are calculated using perceptron
    algorithm.
'''

import random
import sys
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as final_plt  
from matplotlib.pyplot import figure
import os
import math 

weights = [] 
iter_wo_error_impro = 0 # count of epochs without error improvement 
graph_file_number = 1 # used for file name of graphs
save_file = False # used to determine if file should be saved or not
data_file_name = '' # file name where the data is saved
graph_file_name = '' # file name where the graphs are saved
graph_every_correction = False # determines how many times graphs are made
folder_name = '' # name of folder where graphs and results are saved
solve = True # used for solving random data sets or not
total_wrong = 0
wrong = 0
#-----------------------------------------------------------------------------

def evaluate(data, lrn_rate, epoch, curr_epoch, prev_error):
    
    global total_wrong
    global wrong
    
    wrong = 0
    
    if curr_epoch != epoch:
        write_to_file('------------------Error rate for epoch {}: {:.3f}\n'
                      .format(epoch, prev_error))
        curr_epoch = epoch
    
    for row in data:
        h = hypothesis(row)

        if h != row[-1]:
            wrong += 1
            new_weights(row, h, row[-1], lrn_rate)
            if graph_every_correction:
                graph(data)
            write_to_file(str(weights)+'\n')
    
    error_rate = wrong / len(data)
    
    compare_error_rate(prev_error, error_rate, data, epoch)
        
    if wrong == 0:
        graph(data)
        write_to_file('\nTotal of wrong h(x): {}\n'.format(total_wrong))
        write_to_file('\nFinal Weights:\n-\n'+str(weights))
        return
    else:
        total_wrong += wrong
        epoch += 1
        evaluate(data, lrn_rate, epoch, curr_epoch, error_rate)

#-----------------------------------------------------------------------------

# Checks if the error rate has improved, and if it hasn't, it adds 1 to the 
# variable iter_wo_impro. If iter_wo_impro goes over 5, then the algorithm
# stops.

def compare_error_rate(prev_error, curr_error, data, epoch):
    
    global iter_wo_error_impro
    
    change = math.fabs(prev_error - curr_error)
    
    # abs(prev_error - curr_error) > 0.05
    if (prev_error < curr_error) or (change < 0.01):
        iter_wo_error_impro += 1
    else:
        iter_wo_error_impro = 0
        
    if iter_wo_error_impro > 8 or epoch >= 100:
        write_to_file('\nFinal Weights:\n-\n'+str(weights))
        graph(data)
        sys.exit("No improvment in error rate")
    
#-----------------------------------------------------------------------------

# Calculate h(x) using h(x) = w0 + sum(wi, xi)

def hypothesis(row):
    
    hyp = weights[0]
    
    for i in range(1, len(weights), 1):
        hyp += weights[i] * row[i-1]
    
    if hyp >= 0.0:
        return 1
    else:
        return 0
    
#-----------------------------------------------------------------------------

# If c(x) != h(x) then new weights are calculated using:
# wi = wi + n * [c(x) - hx()] * xi
# If wi = w0, then formula used is w0 = w0 + n * [c(x) - h(x)]

def new_weights(data, hyp, cla, lrn_rate):
        
    weights[0] = calc_weight0(weights[0], hyp, cla, lrn_rate)
    
    for i in range(1, len(weights), 1):
        weights[i] = round(weights[i] + lrn_rate * (cla - hyp) * data[i-1], 2)
        
    #for row in data:
    #   weights[1] = round(weights[1] + lrn_rate * (cla - hyp) * data[0], 1)
    #   weights[2] = round(weights[2] + lrn_rate * (cla - hyp) * data[1], 1)
    
    return weights

#-----------------------------------------------------------------------------

# Calculates w0

def calc_weight0(weight0, hyp, cla, lrn_rate):
    
    return round(weight0 + lrn_rate * (cla - hyp), 2)

#-----------------------------------------------------------------------------

# Opens are reads a file and converts it to a 2-d list of floats

def file_to_data(file_name):
    
    open_file = open(file_name, "r")

    file = open_file.readlines()
    
    open_file.close()
    
    # takes out new line from data set
    for i in range(len(file)):
        file[i] = file[i].replace('\n', '')

    data = []
    
    #creates 2-d array
    for i in file:
        row = i.split(' ')
        data.append(row)
    
    #converts strings in array to floats
    for row in data:
        for i in range(len(row)):
            row[i] = float(row[i])

    return data

#-----------------------------------------------------------------------------

def write_to_file(strg):
    if save_file:
        file = open(data_file_name, "a")
        file.write(strg)
        file.close()
    
#-----------------------------------------------------------------------------
    
# Plots a single point and colors it blue if the classifier is 1 and
# red if the classifier is 0.

def one_point(row):
    
    if int(row[-1]) == 1:
        plt.scatter(row[0], row[1], label='True', color='blue', s=1)
    else:
        plt.scatter(row[0], row[1], label='False', color='red', s=1)
        
#-----------------------------------------------------------------------------

# Finds the maximum and minimum of the x and y domain
# Plots each point.
# Calculates the slope and y intercept of weights.
# Plots the line and saves the image to a png file.

def graph(data):
    
    global graph_file_number
    correction = total_wrong + wrong
    max_graph_x = min_graph_x = data[0][0]
    max_graph_y = min_graph_y = data[0][1]
    
    plt.clf()
    
    # plots each individual point
    # finds the max and min for the domain and range of the data 
    for row in data:
        
        if max_graph_x < row[0]:
            max_graph_x = row[0]
            
        if min_graph_x > row[0]:
            min_graph_x = row[0]
            
        if max_graph_y < row[1]:
            max_graph_y = row[1]
            
        if min_graph_y > row[1]:
            min_graph_y = row[1]
            
        one_point(row)
        
    if weights[2] != 0:        
        m = weights[1] / -weights[2]
        b = weights[0] / -weights[2]
        
        if (math.fabs(min_graph_x) + math.fabs(max_graph_x)) > (math.fabs(min_graph_y) + math.fabs(max_graph_y)):
            plt.plot(math.fabs(min_graph_x) + math.fabs(max_graph_x))
        else:
            plt.plot(math.fabs(min_graph_y) + math.fabs(max_graph_y))
            
        plt.xlim(min_graph_x-1, max_graph_x+1)
        plt.ylim(min_graph_y-1, max_graph_y+1)
        plt.gca().set_aspect('equal', adjustable='box')
        
        x = np.linspace(min_graph_x, max_graph_x)
        plt.plot(x, m*x+b, linestyle='solid', color='black')
        
        
        plt.title('0 = {} + ({:.1f})x1 + ({:.1f})x2           Weight Correction {}'
                  .format(weights[0], weights[1], weights[2], correction))
    
    if graph_every_correction:
        new_graph_file_name = graph_file_name+'_'+str(graph_file_number)
        graph_file_number += 1
        plt.savefig(new_graph_file_name, dpi=650)
    else:
        plt.savefig(graph_file_name, dpi=650)
    

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Writes a random data set size, domain, solution, and data points to a file.
def random_write_data_set(data, file_name, maximum, minimum, size):
    file = open(file_name, "w")
    
    file.write('Data set size: {}\n'.format(size))
    file.write('Data set domain: [{}, {}]\n'.format(minimum, maximum))
    
    file.write('Solution:\n')
    file.write(str(weights[0])+' ')
    file.write(str(weights[1])+' ')
    file.write(str(weights[2])+'\n\n')
    
    for row in data:
        for i in range(len(row)):
            if i != 2:
                file.write(str(row[i])+' ')
            else:
                file.write(str(row[i]))
        file.write('\n')
        
    file.close()

#-----------------------------------------------------------------------------

# Graphs a random data set with the solution that was used to make the set.

def graph_random_data(data, minimum, maximum):
    
    plt.clf()
    
    solution_graph_name = folder_name+'_solution'
    
    # plots each individual point
    for row in data:        
        one_point(row)
        
    if weights[2] != 0:        
        m = weights[1] / -weights[2]
        b = weights[0] / -weights[2]
    
        x = np.linspace(minimum, maximum)
        plt.plot(x, m*x+b, linestyle='solid', color='black')
        
    plt.xlim(minimum-10, maximum+10)
    plt.ylim(minimum-10, maximum+10)
    plt.gca().set_aspect('equal', adjustable='box')
        
    plt.title('0 = {} + ({:.1f})x1 + ({:.1f})x2'
                  .format(weights[0], weights[1], weights[2]))
    plt.savefig(solution_graph_name, dpi=650)

#-----------------------------------------------------------------------------
    
# Creates a random data set by randomly generating numbers from [min, max],
# and plugging them into an equation with randomly generated weights from
# [-1, 1]. Then classifies the data pair by using random_data_set_classifier().

def random_data_set(size, minimum, maximum):
    
    data = []
    
    for i in range(size):
        one_row = []
        one_row.append(round(random.uniform(minimum, maximum), 2))
        one_row.append(round(random.uniform(minimum, maximum), 2))
        if random_data_set_classifier(one_row) == 1:
            one_row.append(1)
        else:
            one_row.append(0)
        data.append(one_row)
            
    return data

#-----------------------------------------------------------------------------

# Determines the class of a random data pair by plugging it into the linear
# equation and sending back 1 if it's positive and 0 if it's negative.

def random_data_set_classifier(row):
    
    cla = weights[0] + (weights[1] * row[0]) + (weights[2] * row[1])
    
    if cla >= 0.0:
        return 1
    else:
        return 0

#-----------------------------------------------------------------------------

# Creates folder
    
def make_directory(folder_name):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print("Directory " , folder_name ,  " already exists")
        
#-----------------------------------------------------------------------------

# Checks that c(x) = h(x) for final weights

def check(data):
    file = open(folder_name+'_check', 'w+')
    
    file.write('c | h\n_____\n')
    for row in data:
        h = hypothesis(row)
        file.write(str(row[-1]) + ' | ' + str(h) + '\n')
        if row[-1] != h:
            print('---------ERROR---------\nc(x) != h(x) for: ')
            print(row, row[-1], h)
    
    
    
#-----------------------------------------------------------------------------

lrn_rate = float(input('Enter learning rate: '))

# Makes folder to store files in

folder_name = input('Enter name of folder to hold files: ')
folder_name = '/Users/austinschladant/Perceptron_Algorith/'+folder_name+'/'
make_directory(folder_name)
    
# input = r: creates linearly seperable data set with parameters size, min,
# and max. Saves the data set, solution, size, minimum, and maximum. Graphs 
# the data set with the solution. Asks if you want to solve data set or not.
#
# input = f: loads data set

if input('(r)andom data set or (f)ile: ') == 'r':
    
    # solution of random data set
    
    weights.append(round(random.uniform(-1,1), 2))
    weights.append(round(random.uniform(-1,1), 2))
    weights.append(round(random.uniform(-1,1), 2))
        
    solve = False
    size = int(input('Enter size of set: '))
    minimum = int(input('Enter minimum domain value: '))
    maximum = int(input('Enter maximum domain value: '))
    data = random_data_set(size, minimum, maximum)
    
    if input('Save random data set? (y): ') == 'y':
        file_name = folder_name+'_data'
        random_write_data_set(data, file_name, maximum, minimum, size)
    if input('Graph points with solution line? (y): ') == 'y':
        graph_random_data(data, minimum, maximum)
    if input('Solve data set? (y)') == 'y':
        solve = True
else:
    load_file_name = input('Enter file name: ')
    load_file_name = '/Users/austinschladant/Perceptron_Algorith/'+load_file_name+'.txt'
    data = file_to_data(load_file_name)
    
# input = y: saves the result of each weight change, the number of times the
# weights were changed, and the number of epochs.

if input('Save results? (y): ') == 'y':
    save_file = True
    data_file_name = folder_name+'_results'
    file = open(data_file_name, "w+")
    file.write('Learning Rate: {}\n\n'.format(lrn_rate))
    file.close()
    
# input = y: makes a new png file of the line and points everytime the weights
# change.
#
# input != y: asks if you want to save the points and solution.

if input('Graph after every new weight calculation? (y): ') == 'y':
    graph_every_correction = True
    graph_file_name = folder_name+'_graph'
else:
    input('Save final line? (y): ') == 'y'
    graph_file_name = folder_name+'_graph'

if solve:
    
    weights = [] # reset list because of random ds solution
    
    # creates a list random weights from [-1,1]
    
    for x in data[0]:
        weights.append(round(random.uniform(-1,1), 2))
    
    write_to_file(str(weights)+'\n')
    
    evaluate(data, lrn_rate, 0, 0, 0)
    
    check(data)
        
    
