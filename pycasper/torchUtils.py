import torch

class CustomTensor(torch.Tensor):
  def split_dims(self, split_size, dim=0):
    assert isinstance(split_size, int), 'split_size={} is not an int'.format(split_size)
    assert not self.shape[dim]%split_size, 'dim={} is not divisible by split_size={}'.format(dim, split_size)

    new_shape = list(self.shape)
    factor = int(new_shape[dim]/split_size)
    new_shape[dim] = split_size
    new_shape = torch.Size(new_shape[:dim] + [factor] + new_shape[dim:])
    return CustomTensor(self.clone().view(new_shape))

  def merge(self, dim=0, dim2=1):
    dim = dim + len(self.shape) if dim<0 else dim
    dim2 = dim2 + len(self.shape) if dim2<0 else dim2
    
    assert dim<dim2 or (), 'dim2={} must be bigger than dim={}'.format(dim2, dim)
    
    permutation = list(range(len(self.shape)))
    permutation = permutation[:dim + 1] + permutation[dim2:dim2+1] + permutation[dim+1:dim2] + permutation[dim2+1:]
    
    output = self.clone().permute(permutation)
    new_shape = list(output.shape)
    new_shape = torch.Size(new_shape[:dim] + [new_shape[dim]*new_shape[dim+1]] + new_shape[dim+2:])
     
    return CustomTensor(output.contiguous().view(new_shape))
