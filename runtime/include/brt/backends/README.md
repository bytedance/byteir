# BRT Backends

## How to add a new backend
First, create a unique folder for a given HW backend.

Each HW backend folder typically has ```device``` and ```providers``` folders. 
The ```device``` folder contains the HW-specific classes, and the derived runtime abstract classes, such as ```Allocator``` and ```WorkQueue```.
The ```providers``` folder contains at least one ```provider```, which represents a collection of op implementations. 
Note a backend can have more than one ```provider```. For example, a ```CUDA``` backend can have one ```provider``` relying on custom libraries and another ```provider``` using TensorRT.
Muliple ```providers``` can work together to execute a model. 
Note if multiple ```providers``` all have implmentations for a given op, the op will be executed by the **first** (in terms of the registration order) ```provider```.


### Device
```Allocator``` is a class allocating memory for the specific HW. 
Note multiple cards can be distinguished from ```device_id```. 
The ```bfc_arena``` can be applied to a base ```Allocator``` to form an ```arena``` allocator using a bfc algorithm.

```WorkQueue``` is a class modeling a command queue for the specific HW. 
Tasks of a given HW will be enqueued in order based on the predefined contract of the compiler.
Note the predefined contract may or may not always imply the real execution order, but the simplest contract of the compiler uses execution order. 

### Provider
A ```provider``` typically implements a collection of op implementations, each of which is derived from ```OpKernel```, and register during construction of the ```provider```.

```OpKernel``` can specify execution logic for construction (coded in constructor), ```ProloguePerSession```, ```EpiloguePerSession```, ```ProloguePerFrame```, ```EpiloguePerFrame```, and ```Run```.
The construciton happens per ```OpKernel``` instance; 
```ProloguePerSession``` executes in the beginning of a session; 
```EpiloguePerSession``` executes in the end of a session; 
```ProloguePerFrame``` execute in the beginning of a frame. Here, a frame is defined as an I\O workspace of inference or training; 
```EpiloguePerFrame``` executes in the end of a frame;
```Run``` executes per run. 
Note if a frame is reused, say run twice, ```ProloguePerFrame``` and ```EpiloguePerFrame``` only execute once, while ```Run``` would execute twice.



