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


class ASYMMETRICPC(PostQuantizer):
    """
    Asymmetric, per-Channel Quantizer
    """

    def __init__(self):
        """
        Left for later changes in later adaptations of the quantizer
        """
        super(ASYMMETRICPC, self).__init__(quant_type=Constants.QZ_ASYMMETRICPC)

    def quantize(self, some_tensor, bits, shape, depthwise=False) -> dict:
        """Quantize weights
        :params some_tensor: an ndarray (numpy array)
        :returns dequant_array: an ndarray quantized, same dtype as some_tensor
        """

        self.numbits = bits

        assert some_tensor.dtype == np.float32, "Weight array not fp32"
        some_tensor = np.reshape(some_tensor, shape)
        quant_arr = np.copy(some_tensor)
        dequant_array = np.copy(some_tensor)
        rang = ((1 << bits) - 1)

        if len(some_tensor.shape) == 4: # Check if this is a Conv tensor
            # For Conv2D, quantize along the first axis
            if depthwise == False:
                scale = [0] * some_tensor.shape[0]
                zpoint = [0] * some_tensor.shape[0]
                max_val = [0] * some_tensor.shape[0]
                min_val = [0] * some_tensor.shape[0]
                for i in range(quant_arr.shape[0]):
                    max_val[i], min_val[i] = np.amax(some_tensor[i,:,:,:]), np.amin(some_tensor[i,:,:,:])
                    # Ensure zero is included in the range, see page 4-5 of https://arxiv.org/pdf/1806.08342.pdf
                    min_val[i] = min(min_val[i], 0.0)
                    max_val[i] = max(max_val[i], 0.0)
                    scale[i] = (max_val[i] - min_val[i]) / rang
                    if scale[i] == 0.0:
                        scale[i] = 1.0
                        zpoint[i] = 0
                    else:
                        zpoint[i] = np.around(-np.divide(min_val[i], scale[i]))
                    quant_arr[i,:,:,:] = np.around(np.divide(some_tensor[i,:,:,:], scale[i]) + zpoint[i])
                    dequant_array[i,:,:,:] = np.multiply(scale[i], (quant_arr[i,:,:,:] - zpoint[i]))

            # For DepthwiseConv2D, quantize along the last axis
            elif depthwise:
                scale = [0] * some_tensor.shape[-1]
                zpoint = [0] * some_tensor.shape[-1]
                max_val = [0] * some_tensor.shape[-1]
                min_val = [0] * some_tensor.shape[-1]
                for i in range(quant_arr.shape[-1]):
                    max_val[i], min_val[i] = np.amax(some_tensor[:,:,:,i]), np.amin(some_tensor[:,:,:,i])
                    # Ensure zero is included in the range, see page 4-5 of https://arxiv.org/pdf/1806.08342.pdf
                    min_val[i] = min(min_val[i], 0.0)
                    max_val[i] = max(max_val[i], 0.0)
                    scale[i] = (max_val[i] - min_val[i]) / rang
                    if scale[i] == 0.0:
                        scale[i] = 1.0
                        zpoint[i] = 0
                    else:
                        zpoint[i] = np.around(-np.divide(min_val[i], scale[i]))
                    quant_arr[:,:,:,i] = np.around(np.divide(some_tensor[:,:,:,i], scale[i]) + zpoint[i])
                    dequant_array[:,:,:,i] = np.multiply(scale[i], (quant_arr[:,:,:,i] - zpoint[i]))

        else: # Tensor is not Conv2D, quantize per-tensor
            max_val, min_val = np.amax(some_tensor), np.amin(some_tensor)
            # Ensure zero is included in the range, see page 4-5 of https://arxiv.org/pdf/1806.08342.pdf
            min_val = min(min_val, 0.0)
            max_val = max(max_val, 0.0)
            scale = (max_val - min_val) / rang
            if scale == 0.0:
                scale = 1.0
                zpoint = 0
            else:
                zpoint = np.around(-np.divide(min_val, scale))
            quant_arr = np.around(np.divide(some_tensor, scale) + zpoint)
            dequant_array = np.multiply(scale, (quant_arr - zpoint))
            max_val, min_val, scale, zpoint = [max_val], [min_val], [scale], [zpoint]

        # convert all params (some may be lists) to np arrays
        scale = np.array(scale)
        zpoint = np.array(zpoint)
        max_val = np.array(max_val)
        min_val = np.array(min_val)

        # sanity check
        assert quant_arr.dtype == np.float32, "Quantizer is casting"
        assert quant_arr.shape == some_tensor.shape, "Quantizer is reshaping"

        # flatten arrays to match incoming shape
        quant_arr = np.ndarray.flatten(quant_arr)
        quant_arr = np.clip(quant_arr, 0, rang)
        assert(np.amin(quant_arr) >= 0), "asymmetric-pc should not have negative values"
        assert(np.amax(quant_arr) <= (2**bits)-1), "asymmetric-pc value too large for bits {}".format(bits)
        dequant_array = np.ndarray.flatten(dequant_array)

        return self.create_qnn_params(min_val, max_val, self.numbits, dequant_array, scale, zpoint, quant_arr)

    def number_of_bits(self):
        """
        Asymmetic quantizer is already defined with bits

        :returns the number of bits after quantization for in_tensor
        """
        return self.numbits

    def quantization_error(self, some_tensor, bits=32):
        """
        Function to compute the quantization error
        """
        quantized_obj = self.quantize(some_tensor, bits=bits)
        return np.linalg.norm(np.reshape(some_tensor, [-1]) - np.reshape(quantized_obj["dequant_array"], [-1]))


class SYMMETRIC(PostQuantizer):
    """
    Symmetric Quantizer
    """

    def __init__(self):
        """
        Left for later changes in later adaptations of the quantizer
        """
        super(SYMMETRIC, self).__init__(quant_type=Constants.QZ_SYMMETRIC)

    def quantize(self, some_tensor, bits=32) -> dict:
        """Quantize weights
        :params some_tensor: an ndarray (numpy array)
        :returns dequant_array: an ndarray quantized, same dtype as some_tensor
        """
        self.numbits = bits

        assert some_tensor.dtype == np.float32, "Weight array not fp32"
        # Input array max and min (not channel-wise)
        max_abs_val = max(abs(np.amax(some_tensor)), abs(np.amin(some_tensor)))
        max_val, min_val = max_abs_val, -max_abs_val
        # range of the quantizer
        rang = ((1 << (bits-1)) - 1)
        # scale and bias-point of quantization
        scale = max_abs_val / rang
        if scale == 0.0:
            scale = 1.0
        zpoint = 0

        # Quantize
        quant_arr = np.around(np.divide(some_tensor, scale))
        # sanity check
        assert quant_arr.dtype == np.float32, "Quantizer is casting"
        assert quant_arr.shape == some_tensor.shape, "Quantizer is reshaping"

        # Dequantize the array for sending back
        dequant_array = np.multiply(scale, quant_arr)

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


class SYMMETRICPC(PostQuantizer):
    """
    Symmetric, per-Channel Quantizer
    """

    def __init__(self):
        """
        Left for later changes in later adaptations of the quantizer
        """
        super(SYMMETRICPC, self).__init__(quant_type=Constants.QZ_SYMMETRICPC)

    def quantize(self, some_tensor, bits, shape, depthwise=False) -> dict:
        """Quantize weights
        :params some_tensor: an ndarray (numpy array)
        :returns dequant_array: an ndarray quantized, same dtype as some_tensor
        """

        self.numbits = bits

        assert some_tensor.dtype == np.float32, "Weight array not fp32"
        some_tensor = np.reshape(some_tensor, shape)
        quant_arr = np.copy(some_tensor)
        dequant_array = np.copy(some_tensor)
        rang = ((1 << (bits-1)) - 1)

        if len(some_tensor.shape) == 4: # Check if this is a Conv tensor
            # For Conv2D, quantize along the first axis
            if depthwise == False:
                scale = [0] * some_tensor.shape[0]
                max_abs_val = [0] * some_tensor.shape[0]
                max_val = [0] * some_tensor.shape[0]
                min_val = [0] * some_tensor.shape[0]
                zpoint = [0] * some_tensor.shape[0]
                for i in range(quant_arr.shape[0]):
                    max_abs_val[i] = max(abs(np.amax(some_tensor[i,:,:,:])), abs(np.amin(some_tensor[i,:,:,:])))
                    max_val[i], min_val[i] = max_abs_val[i], -max_abs_val[i]
                    scale[i] = max_abs_val[i] / rang
                    if scale[i] == 0.0:
                        scale[i] = 1.0
                    zpoint[i] = 0
                    quant_arr[i,:,:,:] = np.around(np.divide(some_tensor[i,:,:,:], scale[i]))
                    dequant_array[i,:,:,:] = np.multiply(scale[i], quant_arr[i,:,:,:])

            # For DepthwiseConv2D, quantize along the last axis
            elif depthwise:
                scale = [0] * some_tensor.shape[-1]
                max_abs_val = [0] * some_tensor.shape[-1]
                max_val = [0] * some_tensor.shape[-1]
                min_val = [0] * some_tensor.shape[-1]
                zpoint = [0] * some_tensor.shape[-1]
                for i in range(quant_arr.shape[-1]):
                    max_abs_val[i] = max(abs(np.amax(some_tensor[:,:,:,i])), abs(np.amin(some_tensor[:,:,:,i])))
                    max_val[i], min_val[i] = max_abs_val[i], -max_abs_val[i]
                    scale[i] = max_abs_val[i] / rang
                    if scale[i] == 0.0:
                        scale[i] = 1.0
                    zpoint[i] = 0
                    quant_arr[:,:,:,i] = np.around(np.divide(some_tensor[:,:,:,i], scale[i]))
                    dequant_array[:,:,:,i] = np.multiply(scale[i], quant_arr[:,:,:,i])

        else: # Tensor is not Conv2D, quantize per-tensor
            max_abs_val = max(abs(np.amax(some_tensor)), abs(np.amin(some_tensor)))
            max_val, min_val = max_abs_val, -max_abs_val
            scale = max_abs_val / rang
            if scale == 0.0:
                scale = 1.0
            zpoint = 0
            quant_arr = np.around(np.divide(some_tensor, scale))
            dequant_array = np.multiply(scale, quant_arr)
            max_val, min_val, scale, zpoint = [max_val], [min_val], [scale], [zpoint]

        # convert all params (some may be lists) to np arrays
        scale = np.array(scale)
        max_val = np.array(max_val)
        min_val = np.array(min_val)

        # sanity check
        assert quant_arr.dtype == np.float32, "Quantizer is casting"
        assert quant_arr.shape == some_tensor.shape, "Quantizer is reshaping"

        # flatten arrays to match incoming shape
        quant_arr = np.ndarray.flatten(quant_arr)
        dequant_array = np.ndarray.flatten(dequant_array)

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


class POWER_OF_TWO(PostQuantizer):
    """'Power of Two' Quantization (as in GTC)
    Implements the logic as in GTCQuantization without trainable params
    """

    def __init__(self):
        """
        Left for later changes to the adaptations of this quantizer
        """
        super(POWER_OF_TWO, self).__init__(quant_type=Constants.QZ_POWER_OF_TWO)

    def quantize(self, some_tensor, bits=32) -> dict:
        """Quantize the weights
        :params some_tensor : an ndarray (numpy array)
        :returns quant_array : Power-of-two quantized array (np array)
        """
        # sanity check
        assert some_tensor.dtype == np.float32, "Weight array not fp32"
        max_val, min_val = np.amax(some_tensor), np.amin(some_tensor)
        # # initilize quant array
        # quant_arr = np.zeros(some_tensor.shape)
        # find the exponents
        rounded_exps = np.around(np.log2(abs(some_tensor) + _epi))
        # find the number of bits quantized to
        self.numbits = self.number_of_bits(rounded_exps)
        # Quantize array
        quant_arr = self.sign_with_zero(some_tensor) * (2. ** rounded_exps)
        return self.create_qnn_params(min_val, max_val, self.numbits, quant_arr, 1.0, int(0), quant_arr)
        #return quant_arr

    @staticmethod
    def sign_with_zero(some_tensor):
        """
        calculate piece-wise constant sign that is zero for values in
        vicinity of zero +- epi (1e-6), otherwise returns sign
        :params some_tensor : an ndarray (numpy array)
        :returns sign_mask : a binary mask with signs (+1, -1, 0)
        """

        # pos and neg mask are added both at the end
        pos_mask = np.where(np.greater_equal(some_tensor, _epi),
                            np.ones_like(some_tensor), np.zeros_like(some_tensor))
        neg_mask = np.where(np.less_equal(some_tensor, _epi),
                            -1 * np.ones_like(some_tensor), np.zeros_like(some_tensor))
        sign_mask = pos_mask + neg_mask

        return sign_mask

    def number_of_bits(self, rounded_exponents):
        """ Find the quantized number of bits after power=of-two
        :params rounded_exponents: array of exponents of 2
        :returns numbits: the number of quantized levels
        """
        # min, max exponent in the quantized array
        min_exp, max_exp = np.min(rounded_exponents), np.max(rounded_exponents)
        # range of exponent
        rang_exp = max_exp - min_exp + 1
        # return the number of bits
        return (1 + np.ceil(np.log2(rang_exp)))

    # TODO: is this correct?
    def quantization_error(self, some_tensor, bits=32):
        """
        Function to compute the quantization error
        """
        quantized_obj = self.quantize(some_tensor, bits=bits)
        return np.linalg.norm(np.reshape(some_tensor, [-1]) - np.reshape(quantized_obj["dequant_array"], [-1]))


class Stochastic(object):

    # def __init__(self):
    #     super(Stochastic, self).__init__(quant_type="Stochastic")

    def compute_prob(self, bi, bj, collapsed_tensor):
        """ Compute the Joint Probability

        """
        pass

    def _get_bins(self, some_tensor):
        """ get the bins for quantizing

        :params some_tensor: a tensor/ndarray
        """
        max_bin, min_bin = np.around(np.amax(some_tensor)), \
                           np.around(np.amin(some_tensor))

        bin_array = np.arange(min_bin - 1, max_bin, 0.5)
        return bin_array, bin_array.shape[0]

    def quantize(self, some_tensor) -> dict:
        """Stochastic quantization
        """
        raise NotImplementedError('Stochastic Yet to come')
        # original_shape = np.shape(some_tensor)
        # flat_tensor = np.reshape(some_tensor, (-1))

        # bins, num_bins = self._get_bins(flat_tensor)

        # flat_tensor = np.clip(flat_tensor, bins[0] + EPSILON, bins[-1] - EPSILON)

        # X = np.expand_dims(np.expand_dims(flat_tensor, axis=-1), axis=-1)
        # Prob = np.zeros((flat_tensor.shape[0], num_bins, num_bins))

        # bi = np.expand_dims(np.expand_dims(bins, axis=0), axis=-1)
        # bj = np.expand_dims(np.expand_dims(bins, axis=0), axis=0)

        # for i, bi in enumerate(bins):
        #     j = i + 1
        #     while j <= num_bins:
        #         inds = np.where((X >= bi) * (X < bins[j]))
        #         j += 1
        #         pdb.set_trace()
        # Prob[inds] = 1 - ((X[inds] - bi) / (bj - bi))
        # Pij = 1 - ((X - bi) / (bj - bi))

        # return reduced_jp, bins

# def rounding(self, some_tensor):
#     """ Stochastic rounding attr

#     Overrides the rounding in the standalone quantizers
#     :params some_tensor: A tensor/ndarray
#     :return some_tensor: Stochastically rounded quantized tensor/ndarray
#     """
#     return np.around(some_tensor)


# def get_bins(self, some_tensor):
#     """ Stochastic rounding attr

#     method to get the extreme bins for stochastic rounding
#     :params some_tensor: A tensor/ndarray
#     :return nbins: an array of bins for stochastic rounding
#     """
#     max_bin = np.around(np.amax(some_tensor))
#     min_bin = np.around(np.amin(some_tensor))
#     nbins = np.arange(min_bin - 1, max_bin + 0.5, 1)

#     return nbins
