import struct
import sys

endianness_map = {
    '>': 'be',
    '<': 'le',
    '=': sys.byteorder,
    '|': 'not applicable',
}

dtype_map = {
    'int16': ('int', 16),
    'int32': ('int', 32),
    'int64': ('int', 64),
    'float16': ('float', 16),
    'float32': ('float', 32),
    'float64': ('float', 64)
}


def get_endianness(arr):
    dtype = arr.dtype.newbyteorder('S').newbyteorder('S')
    return endianness_map[dtype.byteorder]


def export_to_wav(path, arr, output_rate, channels):
    dtype, bits = dtype_map[str(arr.dtype)]

    if dtype == 'float':
        sample_width = bits // 8
        byte_count = len(arr) * sample_width  # for example 32-bit floats
        wav_file = ""

        # write the header
        wav_file += struct.pack('<ccccIccccccccIHHIIHH',
                                'R', 'I', 'F', 'F',
                                byte_count + 0x2c - 8,  # header size
                                'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                                0x10,  # size of 'fmt ' header
                                3,  # format 3 = floating-point PCM
                                channels,  # channels
                                output_rate,  # samples / second
                                output_rate * sample_width,  # bytes / second
                                sample_width,  # block alignment
                                sample_width * 8)  # bits / sample
        wav_file += struct.pack('<ccccI',
                                'd', 'a', 't', 'a', byte_count)

        for sample in arr:
            wav_file += struct.pack("<f", sample)

        with open(path, 'wb') as f:
            f.write(wav_file)
    else:
        raise NotImplementedError

