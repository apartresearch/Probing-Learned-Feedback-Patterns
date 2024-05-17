from torch import Tensor
class ModelCustomizer:
    '''
    Used to customize model layer numbers and other network parsing details
    '''

    def __init__(self):
        '''
        Initialize
        '''
        self.target_layers = None

    def set_target_layers(self) -> list[str]:
        '''
        Set target layers
        '''

    def get_target_layers(self) -> list[str]:
        '''
        Get target layers.
        '''

    def parse_layer_name_to_layer_number(self, layer_name) -> str:
        '''
        Parse layer name to layer number
        '''

    def convert_ae_dict_keys(self, autoencoders_dict: [str, Tensor]):
        '''
        Parse ae dict keys to full layer names.
        '''

class GPTNeoCustomizer(ModelCustomizer):

    def get_target_layers(self) -> list[str]:
        if self.target_layers:
            return self.target_layers
        else:
            return [self.layer_num_to_full_name(layer_no) for layer_no in range(12)]

    def set_target_layers(self, target_layers):
        self.target_layers = target_layers

    def layer_num_to_full_name(self, layer_no):
        return f'transformer.h.{layer_no}.mlp'

    def parse_layer_name_to_layer_number(self, layer_name) -> str:
        return layer_name.split('.')[-2]

    # Standardize layer names to full names instead of 'int'
    def convert_ae_dict_keys(self, autoencoders_dict: [str, Tensor]):
        output_dict = {}
        for key, autoencoder in autoencoders_dict.items():
            output_dict[self.layer_num_to_full_name(key)] = autoencoder
        return output_dict

class PythiaCustomizer(ModelCustomizer):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.target_layers = None

    def get_target_layers(self) -> list[str]:
        if self.target_layers:
            return self.target_layers
        else:
            return [self.layer_num_to_full_name(layer_no) for layer_no in range(self.num_layers)]

    def set_target_layers(self, target_layers):
        self.target_layers = target_layers

    def layer_num_to_full_name(self, layer_no):
        return f'gpt_neox.layers.{layer_no}.mlp'

    def parse_layer_name_to_layer_number(self, layer_name) -> str:
        return layer_name.split('.')[-2]

    # Standardize layer names to full names instead of 'int'
    def convert_ae_dict_keys(self, autoencoders_dict: [str, Tensor]):
        output_dict = {}
        for key, autoencoder in autoencoders_dict.items():
            output_dict[self.layer_num_to_full_name(key)] = autoencoder
        return output_dict