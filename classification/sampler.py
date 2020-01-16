class repulsiveSampler(object):

    def __init__(self, repulsive_dataset_type, **kwargs):
        self.type = repulsive_dataset_type
        if self.type in ['FASHIONMNIST', 'KMNIST']:
            assert 'dataloader' in kwargs.keys(), 'dataset not specified.'
            self.dataloader = kwargs['dataloader']
            self.dataloader_iterator = iter(kwargs['dataloader'])
        else:
            raise ValueError('Problem, type {} not managed'.format(self.type))

    def sample_batch(self):
        if self.type in ['FASHIONMNIST', 'KMNIST']:
            try:
                data, _ = next(self.dataloader_iterator)
            except StopIteration:
                self.dataloader_iterator = iter(self.dataloader)
                data, _ = next(self.dataloader_iterator)
            return data
