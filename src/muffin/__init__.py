from muffin.utils import MultiFeatureFusionDataset, DoubleFeatureFusionDataset, MultiFeatureNPZdataset, DoubleFeatureNPZdataset, serialize_doublefeature, serialize_multifeature
from muffin.model import single_feature_model, double_features_model, multi_features_model
from muffin.train import get_model, export, dataloader, train_singlefeature, train_doublefeatures, train_multifeatures, test_model

__version__ = '0.1.0'

