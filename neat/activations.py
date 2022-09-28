# Copyright (c) 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

'''Activation Functions (Will be needed later on probably). Right now 
   tanh is enough.'''

def sigmoid(x):
    
    return 1/(1+math.exp(-1*x))

def modified_sigmoid(x):
    '''Modified Sigmoid as per NEAT paper'''
    return 1/(1+math.exp(-4.9*x))

def tanh(x):
   return math.tanh(x)

def LReLU(x):
    if x >= 0:
        return x
    else:
        return 0.01 * x
