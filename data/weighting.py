import torch


class TiltedDistanceWeight:
    def __init__(self, weight_type, y_scale=0.25):  # 0.25 was derived heuristically.
        self.weight_type = weight_type
        self.y_scale = y_scale

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor), '`tensor` must be a tensor.'
        assert tensor.dim() == 5, '`tensor` is expected to be in the k-space format.'
        device = tensor.device
        height = tensor.size(-3)
        width = tensor.size(-2)
        assert (height % 2 == 0) and (width % 2 == 0), 'Not absolutely necessary but odd sizes are unexpected.'
        mid_height = height / 2
        mid_width = width / 2

        # The indexing might be a bit confusing.
        x_coords = torch.arange(start=-mid_width + 0.5, end=mid_width + 0.5, step=1,
                                device=device).view(1, width).expand(height, width)

        y_coords = torch.arange(start=-mid_height + 0.5, end=mid_height + 0.5, step=1,
                                device=device).view(height, 1).expand(height, width)

        if self.weight_type == 'distance':
            weighting_matrix = torch.sqrt((x_coords ** 2) + self.y_scale * (y_coords ** 2))
        elif self.weight_type == 'root_distance':
            weighting_matrix = torch.sqrt(torch.sqrt((x_coords ** 2) + self.y_scale * (y_coords ** 2)))
        else:
            raise NotImplementedError('Invalid weighting type.')

        weighting_matrix = weighting_matrix.view(1, 1, height, width, 1)

        return weighting_matrix


class SemiDistanceWeight:
    """
    Weighting for semi-k-space. Expects vertical image domain, horizontal k-space data.
    """
    def __init__(self, weight_type):
        self.weight_type = weight_type

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor), '`tensor` must be a tensor.'
        assert tensor.dim() == 5, '`tensor` is expected to be in the k-space format.'
        device = tensor.device
        width = tensor.size(-2)
        assert width % 2 == 0, 'Not absolutely necessary but odd sizes are unexpected.'
        mid_width = width / 2

        x_coords = torch.arange(start=-mid_width + 0.5, end=mid_width + 0.5, step=1, device=device)

        if self.weight_type == 'distance':
            weighting_matrix = torch.abs(x_coords)
        elif self.weight_type == 'root_distance':
            weighting_matrix = torch.sqrt(torch.abs(x_coords))
        else:
            raise NotImplementedError('Invalid weighting type.')

        weighting_matrix = weighting_matrix.view(1, 1, 1, width, 1)

        return weighting_matrix
