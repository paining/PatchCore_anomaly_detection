from collections import OrderedDict
import torch
from torchvision.models import resnet18, resnet50, wide_resnet50_2, wide_resnet101_2
from torchvision.models.resnet import ResNet
# import torch.nn.functional as F

class feature_extractor(torch.nn.Module):
    """
    Feature extractor class.
    We don't need full architecture of pre-trained CNNs.
    We use only front part of its.
    """

    def __init__(self, base_model_type:str = "resnet18",
                 # device:torch.device = torch.device("cuda"),
                 target_layers:list[str]=["layer2","layer3"]):
        super(feature_extractor, self).__init__()
        
        # if not torch.cuda.is_available():
        #     device = torch.device("cpu")

        # load base model
        if base_model_type == "wide_resnet50_2":
            base_model = wide_resnet50_2(pretrained=True, progress=True)
        elif base_model_type == "wide_resnet101_2":
            base_model = wide_resnet101_2(pretrained=True, progress=True)
        elif base_model_type == "resnet50":
            base_model = resnet50(pretrained=True, progress=True)
        else:
            base_model = resnet18(pretrained=True, progress=True)

        dict = OrderedDict(base_model.named_children())
        dict.popitem(last=True)
        dict.popitem(last=True)
        
        self.fe = torch.nn.Sequential(dict)

        base_dict = base_model.state_dict()
        base_dict = {k: v for k, v in base_dict.items() if k in self.fe.state_dict()}
        self.fe.load_state_dict(base_dict)

        self.initialize_features()
        def hook_t(module, input, output):
            self.features.append(output)
        for layer in target_layers:
            self.fe._modules[layer][-1].register_forward_hook(hook_t)

        # # gather preprocess
        # preprocess_list = []
        # for name, module in base_model.named_children():
        #     if name.startswith("layer"):
        #         break
        #     preprocess_list.append( module )
        # self.preprocess = torch.nn.Sequential(*preprocess_list)

        # # gather layers
        # self.layers = OrderedDict()
        # for name, module in base_model.named_children():
        #     if name.startswith("layer"):
        #         self.layers[name] = module
        #         self.add_module(name, module)


        # # gather layers
        # self.layers = OrderedDict()
        # self.upsamples = OrderedDict()
        # i = 0
        # for name, module in base_model.named_children():
        #     if name.startswith("layer"):
        #         i = i+1
        #         self.layers[name] = module
        #         self.upsamples[name] = torch.nn.UpsamplingNearest2d(scale_factor=i)
        
        # save target layers
        self.target_layers = target_layers
        
    def initialize_features(self):
        self.features = []
        self.embeddings = torch.zeros((0,))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.initialize_features()

        self.fe(x)

        # x = self.preprocess(x)
        
        # for name, module in self.layers.items():
        #     x = module(x)
        #     self.features[name] = x
        
        # embedding_list = []
        # for name in self.target_layers:
        #     if name in self.features.keys():
        #         embedding_list.append( self.features[name] )

        # embedding_list = []
        # H, W = 0, 0
        # for name in self.target_layers:
        #     if name in self.features.keys():
        #         embedding_list.append( self.upsamples[name](self.features[name]) )
        #         h, w = self.features[name].shape[2:]
        #         if h > H or w > W:
        #             H, W = h, w

        # torch.cat( embedding_list, dim=1, out = self.embeddings )

        # return self.embeddings

        return self.features
        
    def load_base_model(self, base_model_type:str) -> ResNet:

        if base_model_type == "wide_resnet50_2":
            base_model = wide_resnet50_2(pretrained=True, progress=True)
        elif base_model_type == "wide_resnet101_2":
            base_model = wide_resnet101_2(pretrained=True, progress=True)
        elif base_model_type == "resnet50":
            base_model = resnet50(pretrained=True, progress=True)
        else:
            base_model = resnet18(pretrained=True, progress=True)