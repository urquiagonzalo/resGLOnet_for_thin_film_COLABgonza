import torch

def transfer_matrix_layer(thickness, refractive_index, k, ky, pol):
    '''
    args:
        thickness (tensor): batch size x 1 x 1 x 1
        refractive_index (tensor): batch size x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    else:
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)
    
    T11_R = torch.cos(kx * thickness)
    T11_I = torch.zeros_like(T11_R)
    
    T12_R = torch.zeros_like(T11_R)
    T12_I = torch.sin(kx * thickness) * k / kx * pol_multiplier
    
    T21_R = torch.zeros_like(T11_R)
    T21_I = torch.sin(kx * thickness) * kx / k / pol_multiplier
    
    T22_R = torch.cos(kx * thickness)
    T22_I = torch.zeros_like(T11_R)
    
    return ((T11_R, T11_I), (T12_R, T12_I), (T21_R, T21_I), (T22_R, T22_I))


def transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): 1 x number of frequencies x 1
        ky (tensor): 1 x number of frequencies x number of angles 
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    N = thicknesses.size(-1)
    numfreq = refractive_indices.size(-1)

    T_stack = ((1., 0.), (0., 0.), (0., 0.), (1., 0.))
    for i in range(N):
        thickness = thicknesses[:, i].view(-1, 1, 1, 1)
        refractive_index = refractive_indices[:, i, :].view(-1, numfreq, 1, 1)
        T_layer = transfer_matrix_layer(thickness, refractive_index, k, ky, pol)
        T_stack = matrix_mul(T_stack, T_layer)
        
    return T_stack


def amp2field(refractive_index, k, ky, pol = 'TM'):
    '''
    args:
        refractive_index (tensor): 1 x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): 1 x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    else:
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)

    return ((1., 0), (1., 0.), (-kx / k / pol_multiplier, 0.), (kx / k / pol_multiplier, 0.))


def TMM_solver(thicknesses, refractive_indices, n_bot, n_top, k, theta, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): number of frequencies
        theta (tensor): number of angles
        n_bot (tensor): 1 or number of frequencies
        n_top (tensor): 1 or number of frequencies
        pol (str): 'TM' or 'TE' or 'both'
     
    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    # adjust the format
    n_bot = n_bot.view(1, -1, 1, 1)
    n_top = n_top.view(1, -1, 1, 1)
    k = k.view(1, -1, 1, 1)
    ky = k * n_bot * torch.sin(theta.view(1, 1, -1, 1))

    # transfer matrix calculation
    T_stack = transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol)
    
    # amplitude to field convertion
    A2F_bot = amp2field(n_bot, k, ky, pol)
    A2F_top = amp2field(n_top, k, ky, pol)
    
    # S matrix
    S_stack = matrix_mul(matrix_inv(A2F_top), matrix_mul(T_stack, A2F_bot))
    
    # reflection 
    Reflection = torch.pow(complex_abs(S_stack[2]), 2) / torch.pow(complex_abs(S_stack[3]), 2)
            
    return Reflection