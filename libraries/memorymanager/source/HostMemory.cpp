#include "HostMemory.hpp"

uint8* TraPla::future::HostMemory::getPointer() const
{
    return _host_ptr;
}