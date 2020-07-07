"""Script containing utilities for attaching regularizer
"""
import copy
import json
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  #"3" will suppress every thing, including ERRO messages...
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress TF warning messages.

def apply_quantization(model, verbose=False):
    """A Helper function that applies the quantization 
    based on the regularizer. The idea is to use the 
    regularizer's own quantizer to change the weights.

    Note that this function assumes that "X_regularizer"
    is the regularizer for "X". As long as this convention
    is followed, custom layers can be accommodated.

    T.B.D.: This method may add extra nods to the graph. 
    """
    for layer in model.layers:
        for attribute_name in dir(layer):
            if(attribute_name.lower().find("regularizer") != -1 and\
               attribute_name.lower().find("activity") == -1  
            ):
                ## Identify regularizer
                regularizer = getattr(layer, attribute_name, None)
                if(regularizer is None):
                    print(f"WARNING: \"{attribute_name}\" in \"{layer.name}\" was not found!")
                    continue
                #
                ## Extract the corresponding weight tensor:
                weight_name = attribute_name.replace("_regularizer", "")
                weight_tensor = getattr(layer, weight_name, None)
                if(weight_tensor is None):
                    print(f"WARNING: \"{weight_name}\" in \"{layer.name}\" was not found!")
                    continue
                #
                ## Quantize:
                quantized_weight = tf.keras.backend.eval(
                    regularizer.quantizer(weight_tensor))
                tf.keras.backend.set_value(weight_tensor, quantized_weight) 
                #
                if(verbose):
                    print(f"\tLayer: \"{layer.name}\", Weight: \"{weight_tensor.name}\", Quantizer: \"{regularizer.quantizer_name}\", Bits:{regularizer.num_bits}")

def symmetric_quantizer_error(num_bits, w):
    """Computes the quantization error for symmetric 
    quantization.

    Args:
        num_bits (int): Number of bits.
        w (np.ndarray): Array to measure it quantization 
        error.

    Returns:
        float
    """
    num_bins = float((1 << num_bits) - 1)

    w_min, w_max = np.amin(w), np.amax(w)
    delta = (w_max - w_min)/num_bins
    
    if(delta == 0):
        return 0.0

    q_w = w_min + delta * np.around((w - w_min)/delta)

    return float(np.sum(np.square(w - q_w)))
    # return np.sqrt(np.sum(np.square(w - q_w)) / np.prod(w.shape))

def dynamic_range(w):
    return float(w.max() - w.min())


class QACallback(tf.keras.callbacks.Callback):
    """Callback class to monitor quantization-aware
    training.
    """
    def __init__(self, callables_dict, epoch_begin=False, 
        epoch_end=False, batch_begin=False, batch_end=False):
        """
        Args:
            callables_dict (dict): A dictionary from names to functions to
            apply to the weights.
            epoch_begin (bool): If `True` calls the callables on model
            variables in the beginning of each epoch.
            epoch_end (bool): If `True` calls the callables on model
            variables at the end of each epoch.
            batch_begin (bool): If `True` calls the callables on model
            variables in the beginning of each batch.
            batch_end (bool): If `True` calls the callables on model
            variables at the end of each batch.
        """
        self.callables_dict = callables_dict
        self.epoch_begin = epoch_begin
        self.epoch_end = epoch_end
        self.batch_begin = batch_begin
        self.batch_end = batch_end

    def _run_callables_on_model_variables(self, print_str=None):
        if(print_str):
            print(print_str)
        for var in self.model.trainable_variables:
            if "bias" not in var.name:
                var_array = tf.keras.backend.eval(var)
                for fun_name, fun in self.callables_dict.items():
                    print(f"\t{fun_name}({var.name}) = {fun(var_array)}")

    def on_batch_begin(self, batch, logs=None):
        if(self.batch_begin):
            self._run_callables_on_model_variables(f"\nBatch {batch} begins...")

    def on_batch_end(self, batch, logs=None):
        if(self.batch_end):
            self._run_callables_on_model_variables(f"\nBatch {batch} ended.")

    def on_epoch_begin(self, epoch, logs=None):
        if(self.epoch_begin):
            self._run_callables_on_model_variables(f"\nEpoch {epoch} begins...")

    def on_epoch_end(self, epoch, logs=None):
        if(self.epoch_end):
            self._run_callables_on_model_variables(f"\nEpoch {epoch} ended.")


class QARegularizer(tf.keras.regularizers.Regularizer):
    """Custom regularizer.
    """
    def __init__(self, num_bits=8, lambda_1=1.0, lambda_2=1.0, 
        lambda_3=1.0, lambda_4=1.0, lambda_5=1.0, quantizer_name="asymmetric"):
        """
        Only lambda_2 and lambda_3 are used. Adding these
        for ease of implementation later on in future versions.
        """
        self.num_bits = num_bits
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.lambda_5 = lambda_5
        self.quantizer_name = quantizer_name
        self.tolerance = 0.05
        self.threshold = self.tolerance * (2**self.num_bits - 1)
        #
        self.num_intervals = tf.keras.backend.constant(
            float((1 << self.num_bits) - 1), 
            dtype=tf.float32, name="num_intervals",)


    def asymmetric(self, w,):
        w_min = tf.keras.backend.min(w, axis=None)
        w_max = tf.keras.backend.max(w, axis=None)
        w_delta = (w_max - w_min)/self.num_intervals
        #
        return w_min + w_delta * tf.keras.backend.round(
            tf.math.divide_no_nan(w - w_min, w_delta)
        )

    def quantizer(self, w,):
        return getattr(self, self.quantizer_name)(w)

        
    def __call__(self, w,):
        # quantized_w = self.quantizer(w)
        # tf.keras.backend.print_tensor(quantized_w, message=f"\nQuant {w.name}")
        # tf.keras.backend.print_tensor(w, message=f"{w.name}")
        # quant_loss = self.lambda_2 * tf.keras.backend.sum(
        #     tf.keras.backend.square(w - quantized_w), axis=None)
        mask = tf.keras.backend.less(tf.abs(w),self.threshold)
        l2_mask_loss = self.lambda_3 * tf.keras.backend.sum(
            tf.keras.backend.square(w - tf.cast(mask,'float32')*w), axis=None)
        # l2_loss = self.lambda_3 * tf.keras.backend.sum(
        #     tf.keras.backend.square(w), axis=None)
        #tf.keras.backend.print_tensor(quant_loss, message=f"\nq_loss -- [{w.name}]:")
        #tf.keras.backend.print_tensor(l2_loss, message=f"l2_loss -- [{w.name}]: ")
        # return quant_loss + l2_loss
        return l2_mask_loss
    
    def get_config(self,):
        return {
            'num_bits': int(self.num_bits),
            'lambda_1': float(self.lambda_1),
            'lambda_2': float(self.lambda_2),
            'lambda_3': float(self.lambda_3),
            'lambda_4': float(self.lambda_4),
            'lambda_5': float(self.lambda_5),}

    def get_serialized(self,):
        return tf.keras.regularizers.serialize(self,)


class QAConstraint(tf.keras.constraints.Constraint):
    """QA Constraint to restrict weights to a given 
    range.
    """
    def __init__(self, num_bits=8, tolerance=1.0e-5):
        self.num_bits = num_bits
        self.tolerance = tolerance
        #
        self.num_params = tf.keras.backend.constant(
            value=float((1 << self.num_bits) - 1), 
            dtype=tf.float32, shape=None, name=None)

    def asymmetric_quantizer(self, w,):
        """ T.B.D.: Using tolerance -- we want to 
        replace `round` with something that gives 
        +- tolerance from round.
        """
        w_min = tf.keras.backend.min(w, axis=None)
        w_max = tf.keras.backend.max(w, axis=None)
        w_delta = (w_max - w_min)/self.num_params
        #
        return w_min + w_delta * tf.keras.backend.round(
            tf.math.divide_no_nan(w - w_min, w_delta)
        )

    def __call__(self, w,):
        return self.asymmetric_quantizer(w)

    def get_config(self,):
        return {
            'num_bits': int(self.num_bits),
            'tolerance': float(self.tolerance),}

    def get_serialized(self,):
        return tf.keras.constraints.serialize(self)


def list_tf_keras_model(model_file, **kwargs):
    """Lists the contents of the TF Keras model. This is a CLI
    utility.

    Args:
        model_file (str): Pathname of the saved TF Keras Model.
        **kwargs contains:
            custom_objects (dict): Dictionary of custom layer names
            and their signatures. 
            save_model_json (str):
            save_regularizers_json (str):
    """
    tf.keras.backend.clear_session()
    #
    ## Load model and get the model JSON
    model = tf.keras.models.load_model(
        model_file, custom_objects=None, compile=False)
    #
    model_json = json.loads(model.to_json())
    # print("MODEL JSON: ")
    # print(json.dumps(model_json, indent=4), end="\n\n")
    # if(kwargs.get("save_model_json", None) is not None):
    #     print(f"Saving model JSON to \"{kwargs['save_model_json']}\"...")
    #     with open(kwargs["save_model_json"], "w", encoding="utf-8") as json_file:
    #         json.dump(model_json, json_file, ensure_ascii=False, indent=4)
    #
    ## Compile json of layer regularizers:
    # print("REGULARIZATION & CONSTRAINTS: ")
    layers_regularizer_json = {}
    for layer in model_json["config"]["layers"]:
        class_name = layer["class_name"]
        layer_name = layer["config"]["name"]
        temp_layer_dict = { 
            "class_name": class_name,
            "name": layer_name,}
        for config_name in layer["config"]:
            if(config_name.lower().find("regularizer") != -1 and\
               config_name.lower().find("activity") == -1     or\
               config_name.lower().find("constraint") != -1):
                temp_layer_dict[config_name] = layer["config"][config_name]
        layers_regularizer_json[layer_name] = copy.deepcopy(temp_layer_dict)
    #
    # print(json.dumps(layers_regularizer_json, indent=4), end="\n\n")
    if(kwargs.get("save_regularizers_json", None) is not None):
        # print(f"Saving regularizers JSON to \"{kwargs['save_regularizers_json']}\"...")
        with open(kwargs["save_regularizers_json"], "w", encoding="utf-8") as json_file:
            json.dump(layers_regularizer_json, json_file, ensure_ascii=False, indent=4)
    return layers_regularizer_json
    #
    # ## Print the list of model variables:
    # print("LIST OF VARIABLES: ")
    # for layer in model.layers:
    #     print(layer.name)
    #     for variable in layer.variables:
    #         print(f"\tname:      {variable.name}")
    #         print(f"\tshape:     {variable.shape}")
    #         print(f"\tdtype:     {variable.dtype}")
    #         print(f"\ttrainable: {variable._trainable}", end="\n\n")


def attach_regularizers(tf_keras_model, attch_json_scheme, 
    target_keras_h5_file=None, verbose=True, backend_session_reset=True):
    """Attaches regularizer to tf_keras_model

    Args:
        tf_keras_model (str or tf.keras.Model)
        attch_json_scheme (str or dict)
        verbose (bool): St to `False` to suppress verbosity.

    Returns:
        The new tf keras model instance.
    """
    ## If a model path is provided, 
    if(isinstance(tf_keras_model, str)):
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            tf_keras_model, custom_objects=None, compile=False)
    else:
        model = tf_keras_model

    ## Parse the json if a file or string
    if(isinstance(attch_json_scheme, str)):
        if(attch_json_scheme.split(".")[-1] == "json"):
            with open(attch_json_scheme) as json_file:
                json_scheme = json.load(json_file)
        else:
            json_scheme = json.loads(attch_json_scheme)
    elif(isinstance(attch_json_scheme, dict)):
        json_scheme = attch_json_scheme
    else:
        raise ValueError(f"\"{attch_json_scheme}\" is neither a JSON dict, not a JSON file nor a JSON str!")

    ## Get model JSON and model weights:
    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())
    #
    if(verbose):
        print("SOURCE MODEL JSON:")
        print(json.dumps(model_json, indent=4))
        print("\n\n\n")

    ## Attach regularizers and/or contraints:
    for idx, layer in enumerate(model_json["config"]["layers"]):
        layer_name = layer["config"]["name"]
        if(layer_name not in json_scheme.keys()):
            continue
        for config_name in json_scheme[layer_name]: 
            if(config_name.lower().find("regularizer") != -1 and\
               config_name.lower().find("activity") == -1     or\
               config_name.lower().find("constraint") != -1):
               model_json["config"]["layers"][idx]["config"][config_name] = json_scheme[layer_name][config_name]
    model_json["config"]["name"] = "TA_" + model_json["config"]["name"]  # Attach the prefix "TA_" to model name.
    #
    if(verbose):
        print("NEW MODEL JSON:")
        print(json.dumps(model_json, indent=4))
        print("\n\n\n")

    ## Construct the new model
    if(backend_session_reset):
        tf.keras.backend.clear_session()
    new_model = tf.keras.models.model_from_json(
        json.dumps(model_json), 
        custom_objects={
            "QARegularizer": QARegularizer,
            "QAConstraint": QAConstraint, }
    )
    new_model.set_weights(model_weights)
    #
    if(verbose):
        new_model.summary(print_fn=(lambda *args: print("    ", *args)))
        print("\n\n\n")

    ## Export or return the new model
    if(target_keras_h5_file is not None):
        new_model.save(target_keras_h5_file, overwrite=True, include_optimizer=False, save_format="h5",)
    else:
        return new_model


def main():
    import fire
    fire.Fire({
        "QARegularizer": QARegularizer,
        "QAConstraint": QAConstraint,
        "list": list_tf_keras_model,
        "attach": attach_regularizers, 
    })


if(__name__ == "__main__"):
    main()