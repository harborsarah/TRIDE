import torch
import torch.nn as nn
import torch.linalg as linalg
import math
def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.l1_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.mse_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class binary_cross_entropy(nn.Module):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def forward(self, confidence, radar_gt, mask):
        loss = torch.nn.functional.binary_cross_entropy(confidence, radar_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class smoothness_loss_func(nn.Module):
    def __init__(self):
        super(smoothness_loss_func, self).__init__()
    
    def gradient_yx(self, T):
        '''
        Computes gradients in the y and x directions

        Arg(s):
            T : tensor
                N x C x H x W tensor
        Returns:
            tensor : gradients in y direction
            tensor : gradients in x direction
        '''

        dx = T[:, :, :, :-1] - T[:, :, :, 1:]
        dy = T[:, :, :-1, :] - T[:, :, 1:, :]
        return dy, dx
    
    def forward(self, predict, image):
        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
        
        return smoothness_x + smoothness_y

