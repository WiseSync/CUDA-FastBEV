#pragma once
#include <memory>

namespace fastbev
{
    class Utils
    {

    public:
        template <typename T>
        static T *allocTensorMem(int size)
        {
            T *ptr = nullptr;

#ifdef DEVICE_AM68a
            ptr = TIDLRT_allocSharedMem(64, size);
#else
            ptr = reinterpret_cast<T*>(std::malloc(size));
#endif

            return ptr;
        }
        static void freeTensorMem(void *ptr)
        {
#ifdef DEVICE_AM68a
            TIDLRT_freeSharedMem(ptr);
#else
            std::free(ptr);
#endif
        }
    };

} // namespace fastbev