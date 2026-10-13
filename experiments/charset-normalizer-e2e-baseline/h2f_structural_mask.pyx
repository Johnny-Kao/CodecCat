# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from libc.stdint cimport uint8_t

cdef int BIG5 = 1
cdef int GB18030 = 2
cdef int SHIFT_JIS = 4
cdef int EUC_JP = 8
cdef int EUC_KR = 16
cdef Py_ssize_t LIMIT = 4096

cdef inline bint valid_big5(const uint8_t* b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i = 0
    cdef uint8_t x, y
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if x < 0x81 or x > 0xFE:
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if not ((0x40 <= y <= 0x7E) or (0xA1 <= y <= 0xFE)):
            return False
        i += 2
    return True

cdef inline bint valid_gb18030(const uint8_t* b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i = 0
    cdef uint8_t x, y, z, w
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if x < 0x81 or x > 0xFE:
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if 0x30 <= y <= 0x39:
            if i + 3 >= n:
                return True
            z = b[i + 2]
            w = b[i + 3]
            if not (0x81 <= z <= 0xFE and 0x30 <= w <= 0x39):
                return False
            i += 4
            continue
        if not ((0x40 <= y <= 0x7E) or (0x80 <= y <= 0xFE)):
            return False
        i += 2
    return True

cdef inline bint valid_shift_jis(const uint8_t* b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i = 0
    cdef uint8_t x, y
    while i < n:
        x = b[i]
        if x <= 0x7F or 0xA1 <= x <= 0xDF:
            i += 1
            continue
        if not ((0x81 <= x <= 0x9F) or (0xE0 <= x <= 0xFC)):
            return False
        if i + 1 >= n:
            return True
        y = b[i + 1]
        if not ((0x40 <= y <= 0x7E) or (0x80 <= y <= 0xFC)):
            return False
        i += 2
    return True

cdef inline bint valid_euc_jp(const uint8_t* b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i = 0
    cdef uint8_t x
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if x == 0x8E:
            if i + 1 >= n:
                return True
            if b[i + 1] < 0xA1 or b[i + 1] > 0xDF:
                return False
            i += 2
            continue
        if x == 0x8F:
            if i + 2 >= n:
                return True
            if not (0xA1 <= b[i + 1] <= 0xFE and 0xA1 <= b[i + 2] <= 0xFE):
                return False
            i += 3
            continue
        if x < 0xA1 or x > 0xFE:
            return False
        if i + 1 >= n:
            return True
        if b[i + 1] < 0xA1 or b[i + 1] > 0xFE:
            return False
        i += 2
    return True

cdef inline bint valid_euc_kr(const uint8_t* b, Py_ssize_t n) noexcept:
    cdef Py_ssize_t i = 0
    cdef uint8_t x
    while i < n:
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if x < 0xA1 or x > 0xFE:
            return False
        if i + 1 >= n:
            return True
        if b[i + 1] < 0xA1 or b[i + 1] > 0xFE:
            return False
        i += 2
    return True

cpdef int invalid_mask(bytes data):
    cdef Py_ssize_t n = PyBytes_GET_SIZE(data)
    cdef const uint8_t* b = <const uint8_t*>PyBytes_AS_STRING(data)
    cdef int mask = 0
    if n > LIMIT:
        n = LIMIT
    if not valid_big5(b, n):
        mask |= BIG5
    if not valid_gb18030(b, n):
        mask |= GB18030
    if not valid_shift_jis(b, n):
        mask |= SHIFT_JIS
    if not valid_euc_jp(b, n):
        mask |= EUC_JP
    if not valid_euc_kr(b, n):
        mask |= EUC_KR
    return mask
