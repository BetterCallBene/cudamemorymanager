# Memory Manager

As the name indicates the memory manager has the task to maintain the linear device memory. The library MemoryManager is independent of the TraPla-project. So you can individual separate your memory as it called in containers. For this you need a struct like this:

```cpp
struct MyMemoryStructure
{
    DummyStateContainer dummyStateContainer;
    ...
};
```
to define your memory on host/device. This structure contains the DummyStateContainer with in our case represented all DummyStates. The DummyStates will define as follows:
```cpp
template<typename... args_t>
struct DummyStatesDefinition : public gpu::tuple<args_t...>
{
    // (1) Making base constructor public
    using gpu::tuple<args_t...>::tuple;
    // (2) for the getter function in DummyStateContainer, we need the basetype
    using basetype = gpu::tuple<args_t...>;
    // (3) Problem: Cuda specific constructor will be not public with (1), so we must it redefine
    CUDA_HOSTDEV DummyStatesDefinition(const args_t&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyStatesDefinition(args_t&&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyStatesDefinition(const basetype& rhs)
        : basetype(rhs)
    {
    }
    // (4) calculate the the size of MemoryDefinition
    CUDA_HOSTDEV
    static constexpr uint64 getSize()
    {
        return CalcElementsSize<args_t...>::value;
    }

    // (5) getter/setter function can be individual
    CUDA_HOSTDEV
    auto& getterSetter1()
    {
        return gpu::get<0>(*this);
    }

    CUDA_HOSTDEV
    auto& getterSetter2()
    {
        return gpu::get<1>(*this);
    }
};
// For easy using we can define as the as following. It is only important to use pointers.
using DummyStates = DummyStatesDefinition<float*, int*>;
```
We can now finishing our memory structure with defining our DummyStateContainer with defining it:
```cpp
using DummyStateContainer = MemoryContainer<DummyStates>;
```
Our container can now contain to two rows: One with float and the other one with integer values. About the fact that we can not use dynamic arrays we are needing a assignment function. For propose we defining us follow function:
```cpp
template<>
template<>
const MemoryManager<MyMemoryStructure>::ContainerInfo MemoryManager<MyMemoryStructure>::assignContainer<
    DummyStateContainer>(MyMemoryStructure& memory, DummyStateContainer** container)
{
    // (1) Assign the right pointer address. Important: Here we are using double pointer
    *container = &memory.dummyStateContainer;
    // (2) Fill the container info structure. A unique container id (here can you use a enum as example) and the needed column size.
    return ContainerInfo{firstContainerId, firstContainerSize};
}
```

Additional it is a getter function necessary for our container:

```cpp
template<>
template<>
CUDA_HOSTDEV const MyMemoryStructure* MemoryManagerHostDev<MyMemoryStructure>::getContainer<
    DummyStateContainer>()
{
    // (1) The method getPlatformMemory make the get function independent from where it will be called (host/device).
    return &getPlatformMemory()->dummyStateContainer;
}
```
That's it! We are know defined our memory structure. For easy using we defined us the MemoryManager as following:
```cpp
using DummyManager = MemoryManager<MyMemoryStructure>;
```

Now we can install our memory:

```cpp
DummyManager manager;
auto container = manager.install<DummyStateContainer>();
```
Initialized and get access to our memory.

```cpp
manager.init();
auto element = container.getElementAt(0);
float* firstElement = element.getterSetter1();
int* secondElement = element.getterSetter2();
// If you done with your work, please clean.
manager.cleanUp();
```
For more complex example look at the file ```integration_test.cu```.

## Version 0.1.0

Initial memorymanager functionality.

## ToDo:
* Fix windows build
* Add upload/download interface in install function (Example install<..>(IUploader))
* Add upload/download function in memorymanager like write in the concept project
* Add profiling tests
* Make it more stabil (fix possible crashes)
* Include in productiv code