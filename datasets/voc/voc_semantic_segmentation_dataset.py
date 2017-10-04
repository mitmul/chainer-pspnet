import os

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.datasets.voc import voc_utils


class VOCSemanticSegmentationDataset(VOCSemanticSegmentationDataset):

    """Dataset class for the semantic segmantion task of PASCAL `VOC2012`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_semantic_segmentation_label_names`.
    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val', 'test']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\', '
                '\'test\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
