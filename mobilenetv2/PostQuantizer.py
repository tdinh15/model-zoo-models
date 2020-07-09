#  Copyright (c) 2020 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import abc
import numpy as np
from leip import Constants

_epi = 1e-6
EPSILON = 0.2


class PostQuantizer(abc.ABC):
    """
    Abstract class for post-training Quantizer
    """

    def __init__(self, quant_type: str):
        self.quantizer_type = quant_type

    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def quantize(self, some_tensor, bits=32) -> dict:
        """
        main function to quantize a tensor

        :params some_tensor : A ndarray from the model Variable
                bits : The number of bits to quantize to
        :returns A ndarray with same shape and dtype as in_tensor
        """
        raise NotImplementedError("To be Overidden by derived class")

    @abc.abstractmethod
    def number_of_bits(self) -> int:
        """
        Function to compute the compressed bits for some quantizers

        :returns the number of bits after quantization for in_tensor
        """
        raise NotImplementedError('To be Overidden by the derived class')

    # TODO: are the sub classes implemeting this correctluy?
    @abc.abstractmethod
    def quantization_error(self, in_tensor, bits=32):
        """
        Function to compute the quantization error

        :params in_tensor : A ndarray from the model Variable
        :returns the error in quantization for bit precision
        """
        raise NotImplementedError('To be Overidden for derived class')

    def get_bins(self):
        """ TODO

        """
        raise ("Not Implemented yet")

    @staticmethod
    def create_qnn_params(min_val, max_val, num_bits, dequant_array, scale, zpoint, quant_arr) -> dict:
        # TODO: make this an object later once it works and we can generalize more
        # TODO: min and max do not seem to apply to all quantizers
        qnn_params = dict()
        # these can be used for calibration (TODO: verify this)
        qnn_params['min_value'] = min_val
        qnn_params['max_value'] = max_val
        qnn_params['dequant_array'] = dequant_array
        qnn_params['quant_arr'] = quant_arr
        qnn_params['num_bits'] = num_bits
        # only used for casting to int8
        qnn_params['scale'] = scale
        qnn_params['zero_point'] = zpoint
        return qnn_params


class ASYMMETRIC(PostQuantizer):
    """ASYMMETRIC QUANTIZATION
    Implements the Linear Asymmetric Quantization for given bit precision
    """

    def __init__(self):
        """
        Left for later changes in later adaptations of the quantizer
        """
        super(ASYMMETRIC, self).__init__(quant_type=Constants.QZ_ASYMMETRIC)

    def quantize(self, some_tensor, bits=32) -> dict:
        """Quantize weights
        :params some_tensor: an ndarray (numpy array)
        :returns dequant_array: an ndarray quantized, same dtype as some_tensor
        """
        self.numbits = bits

        assert some_tensor.dtype == np.float32, "Weight array not fp32"
        # Input array max and min (not channel-wise)
        max_val, min_val = np.amax(some_tensor), np.amin(some_tensor)

        # Ensure zero is included in the range, see page 4-5 of https://arxiv.org/pdf/1806.08342.pdf
        min_val = min(min_val, 0.0)
        max_val = max(max_val, 0.0)

        # range of the quantizer
        rang = ((1 << bits) - 1)

        # scale and bias-point of quantization
        scale = (max_val - min_val) / rang
        if scale == 0.0:
            quant_arr = np.around(np.divide(some_tensor, 1.0) + 0)
            dequant_array = np.multiply(1.0, (quant_arr - 0))
            return self.create_qnn_params(min_val, max_val, self.numbits, dequant_array, 1.0, int(0), quant_arr)

        zpoint = np.around(-np.divide(min_val, scale))
        # Quantize
        quant_arr = np.around(np.divide(some_tensor, scale) + zpoint)
        # Gitlab SDK-102:
        # TODO: is this still valid?
        # sanity check
        assert quant_arr.dtype == np.float32, "Quantizer is casting"
        assert quant_arr.shape == some_tensor.shape, "Quantizer is reshaping"

        quant_arr = np.clip(quant_arr, 0, rang)

        # Dequantize the array for sending back
        dequant_array = np.multiply(scale, (quant_arr - zpoint))

        # Gitlab SDK-102, return QNN params
        return self.create_qnn_params(min_val, max_val, self.numbits, dequant_array, scale, zpoint, quant_arr)

    def number_of_bits(self):
        """
        Asymmetric quantizer is already defined with bits

        :returns the number of bits after quantization for in_tensor
        """
        return self.numbits

    def quantization_error(self, some_tensor, bits=32):
        """
        Function to compute the quantization error
        """
        quantized_obj = self.quantize(some_tensor, bits=bits)
        return np.linalg.norm(np.reshape(some_tensor, [-1]) - np.reshape(quantized_obj["dequant_array"], [-1]))
