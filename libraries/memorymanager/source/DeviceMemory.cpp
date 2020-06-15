#include "DeviceMemory.hpp"

uint8* TraPla::future::DeviceMemory::getPointer() const
{
    return _device_ptr;
}