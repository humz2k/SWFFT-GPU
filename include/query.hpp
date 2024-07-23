/**
 * @file query.hpp
 */
#ifndef _SWFFT_QUERY_HPP_
#define _SWFFT_QUERY_HPP_

namespace SWFFT {

template <template <class, class> class T> inline const char* queryName();

template <class T> inline const char* queryName();

} // namespace SWFFT

#endif