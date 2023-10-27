//
// Created by Shujian Qian on 2023-10-12.
//

#ifndef UTIL_CUB_REVERSE_ITERATOR_CUH
#define UTIL_CUB_REVERSE_ITERATOR_CUH

#include <type_traits>

#if (THRUST_VERSION >= 100700)
// This iterator is compatible with Thrust API 1.7 and newer
#    include <thrust/iterator/iterator_facade.h>
#    include <thrust/iterator/iterator_traits.h>
#endif // THRUST_VERSION

namespace epic {

template<typename ValueType, typename InputIteratorT, typename OffsetT = int64_t>
class ReverseIterator
{
private:
    typedef decltype(*std::declval<InputIteratorT>()) deref_type;

public:
    typedef ReverseIterator self_type;
    typedef OffsetT difference_type;
    typedef ValueType value_type;
    typedef ValueType *pointer;
    typedef
        typename std::conditional<std::is_lvalue_reference<deref_type>::value, ValueType &, ValueType>::type reference;

#ifndef THRUST_NS_QUALIFIER
#    define THRUST_NS_QUALIFIER ::thrust
#endif

#if (THRUST_VERSION >= 100700)
    // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
    typedef typename THRUST_NS_QUALIFIER::detail::iterator_facade_category<THRUST_NS_QUALIFIER::any_system_tag,
        THRUST_NS_QUALIFIER::random_access_traversal_tag, value_type,
        reference>::type iterator_category; ///< The iterator category
#else
    typedef std::random_access_iterator_tag iterator_category; ///< The iterator category
#endif // THRUST_VERSION

private:
    InputIteratorT input_itr;

public:
    explicit __host__ __device__ __forceinline__ ReverseIterator(InputIteratorT input_itr)
        : input_itr(input_itr)
    {}

    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        input_itr--;
        return retval;
    }

    __host__ __device__ __forceinline__ self_type operator++()
    {
        input_itr--;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*() const
    {
        return *input_itr;
    }

    template<typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(input_itr - n);
        return retval;
    }

    template<typename Distance>
    __host__ __device__ __forceinline__ self_type operator+=(Distance n)
    {
        input_itr -= n;
        return *this;
    }

    template<typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(input_itr + n);
        return retval;
    }

    template<typename Distance>
    __host__ __device__ __forceinline__ self_type operator-=(Distance n)
    {
        input_itr += n;
        return *this;
    }

    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        /* negative because this iterator is reversed */
        return other.input_itr - input_itr;
    }

    template<typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n) const
    {
        return *(input_itr - n);
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) const
    {
        return input_itr == rhs.input_itr;
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) const
    {
        return input_itr != rhs.input_itr;
    }

    friend std::ostream &operator<<(std::ostream &os, const self_type &itr)
    {
        return os;
    }
};

} // namespace epic

#endif // UTIL_CUB_REVERSE_ITERATOR_CUH
