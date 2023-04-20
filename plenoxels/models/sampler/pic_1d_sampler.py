import torch

class pic_1d_sampler:
    def __init__(self) -> None:
        pass
    
    def __call__(self, grid: torch.nn.parameter.Parameter, coords: torch.Tensor, align_corners=None, mode=None, padding_mode=None) -> torch.Tensor:
        '''
        @param grid:   B x F x W
        @param coords: B x N x 1
        @returns:      B x F x N
        '''
        B, F, W = grid.shape
        N = coords.size(1)

        coords = (coords + 1.) * 0.5 * W
        coords = torch.clamp(coords, min=0, max=W-1-(1e-3)).view(-1)
        
        cords_floor = coords.floor().long()
        weight2 = (coords - cords_floor).view(1, 1, N)
        weight1 = 1. - weight2

        result = grid[..., cords_floor + 1] * weight2 + grid[..., cords_floor] * weight1

        return result.view(B, F, N)


if __name__ == '__main__':
    print(1)