#ifndef TRAPLA_MEMORYMANAGER_HELPER_CPP_14_HPP_
#define TRAPLA_MEMORYMANAGER_HELPER_CPP_14_HPP_

#include "cuda/common.hpp"
#include "device_tuple.hpp"
namespace TraPla
{
namespace future
{
#ifdef _WIN32
    using namespace std;
#else
    using namespace gpu;
#endif

template<typename...>
struct CalcElementsSize : std::integral_constant<uint64, 0>
{
};

template<typename X, typename... Xs>
struct CalcElementsSize<X, Xs...>
    : std::integral_constant<uint64, sizeof(std::remove_pointer_t<X>) + CalcElementsSize<Xs...>::value>
{
};

template<class... Args>
struct DataElementSize
{
    template<std::size_t N>
    CUDA_HOSTDEV static constexpr std::size_t value()
    {
        return sizeof(typename std::remove_pointer<typename tuple_element<N, Args...>::type>::type);
    }
};

template<uint64 N, typename T>
struct CalcIndexPosition
{
    CUDA_HOSTDEV
    static constexpr uint64 value(const uint64 quantity)
    {
        return TraPla::future::DataElementSize<T>::template value<N - 1>() * quantity
               + CalcIndexPosition<N - 1, T>::value(quantity);
    }
};

template<typename T>
struct CalcIndexPosition<1, T>
{
    CUDA_HOSTDEV
    static uint64 value(const uint64 quantity)
    {
        return DataElementSize<T>::template value<0>() * quantity;
    }
};

template<typename T>
struct CalcIndexPosition<0, T>
{
    CUDA_HOSTDEV
    static constexpr uint64 value(const uint64 quantity)
    {
        return 0u;
    }
};
} // namespace future
} // namespace TraPla

#endif // TRAPLA_MEMORYMANAGER_HELPER_CPP_14_HPP_