import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from torchvision import transforms

other_NC = True
pb = True
TPT = None

class graphs:
  def __init__(self):
    self.accuracy     = []
    self.loss         = []
    self.reg_loss     = []

    # NC1
    self.Sw_invSb     = []
    self.Sw = []
    self.Sb = []
    self.Sw_dSb = []

    # NC2
    self.norm_M_CoV   = []
    self.norm_W_CoV   = []
    self.cos_M        = []
    self.cos_W        = []

    # NC3
    self.W_M_dist     = []

    # NC4
    self.NCC_mismatch = []

    # Decomposition
    self.MSE_wd_features = []
    self.LNC1 = []
    self.LNC23 = []
    self.Lperp = []

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

def train(model, criterion, device, num_classes, train_loader, optimizer, epoch, batch_size, debug):
    model.train()

    if pb:
        pbar = tqdm(total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
          loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
          loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())

        loss.backward()
        optimizer.step()

        accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
        
        if pb:
            pbar.update(1)
            pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))

        if debug and batch_idx > 20:
            break

    if pb:
        pbar.close()

# extract penultimate layer features and classifier
class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()   

def analysis(graphs, raw_model, model, criterion_summed, device, num_classes, loader, weight_decay, epoch, debug, loss_name):
    model.eval()

    classifier = raw_model.fc
    classifier.register_forward_hook(hook)

    C = num_classes
    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    for computation in ['Mean','Cov']:
        if pb:
            pbar = tqdm(total=len(loader), position=0, leave=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            output = model(data)
            h = features.value.data.view(data.shape[0],-1) # B CHW

            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                  loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                  loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]

                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

            if pb:
                pbar.update(1)
                pbar.set_description(
                    'Analysis {}\t'
                    'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    computation,
                    epoch,
                    batch_idx,
                    len(loader),
                    100. * batch_idx/ len(loader)))

            if debug and batch_idx > 20:
                break
        if pb:
            pbar.close()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)

    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct/sum(N))
    graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    if other_NC:
        # avg norm
        W  = classifier.weight
        M_norms = torch.norm(M_,  dim=0)
        W_norms = torch.norm(W.T, dim=0)

        graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
        graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # Decomposition of MSE #
    if loss_name == 'MSELoss' and other_NC:

      wd = 0.5 * weight_decay # "\lambda" in manuscript, so this is halved
      St = Sw+Sb
      size_last_layer = Sb.shape[0]
      eye_P = torch.eye(size_last_layer).to(device)
      eye_C = torch.eye(C).to(device)

      St_inv = torch.inverse(St + (wd/(wd+1))*(muG @ muG.T) + wd*eye_P)

      w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
      b_LS = (1/C * torch.ones(C).to(device) - w_LS @ muG.T.squeeze(0)) / (1+wd)
      w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim=1)  # c x n
      b  = classifier.bias
      w_ = torch.cat([W, b.unsqueeze(-1)], dim=1)  # c x n

      LNC1 = 0.5 * (torch.trace(w_LS @ (Sw + wd*eye_P) @ w_LS.T) + wd*torch.norm(b_LS)**2)
      LNC23 = 0.5/C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

      A1 = torch.cat([St + muG @ muG.T + wd*eye_P, muG], dim=1)
      A2 = torch.cat([muG.T, torch.ones([1,1]).to(device) + wd], dim=1)
      A = torch.cat([A1, A2], dim=0)
      Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

      MSE_wd_features = loss + 0.5* weight_decay * (torch.norm(W)**2 + torch.norm(b)**2).item()
      MSE_wd_features *= 0.5

      graphs.MSE_wd_features.append(MSE_wd_features)
      graphs.LNC1.append(LNC1.item())
      graphs.LNC23.append(LNC23.item())
      graphs.Lperp.append(Lperp.item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    Sw = np.sum(Sw)
    Sb = np.sum(Sb)
    graphs.Sw.append(Sw)
    graphs.Sb.append(Sb)
    graphs.Sw_dSb.append(Sw/Sb)

    if other_NC:
        # ||W^T - M_||
        normalized_M = M_ / torch.norm(M_,'fro')
        normalized_W = W.T / torch.norm(W.T,'fro')
        graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

        # mutual coherence
        def coherence(V):
            G = V.T @ V
            G += torch.ones((C,C),device=device) / (C-1)
            G -= torch.diag(torch.diag(G))
            return torch.norm(G,1).item() / (C*(C-1))

        graphs.cos_M.append(coherence(M_/M_norms))
        graphs.cos_W.append(coherence(W.T/W_norms))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot(epoch, cur_epochs, train_graphs, test_graphs, loss_name, no, rd = 4):
    _x = epoch
    plt.figure(1)
    plt.semilogy(cur_epochs, train_graphs.reg_loss) # what's the difference?
    plt.plot(cur_epochs, test_graphs.reg_loss)
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Regularized Loss')
    _y = train_graphs.reg_loss[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = test_graphs.reg_loss[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_reg_loss.png")

    plt.figure(2)
    plt.plot(cur_epochs, 100*(1 - np.array(train_graphs.accuracy)))
    plt.plot(cur_epochs, 100*(1 - np.array(test_graphs.accuracy)))
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Error')
    _y = 100 * (1 - np.array(train_graphs.accuracy))[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = 100 * (1 - np.array(test_graphs.accuracy))[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_error.png")

    plt.figure(3)
    plt.semilogy(cur_epochs, train_graphs.Sw_invSb)
    plt.semilogy(cur_epochs, test_graphs.Sw_invSb)
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Tr{Sw Sb^-1}')
    plt.title('NC1: Activation Collapse')
    _y = train_graphs.Sw_invSb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = test_graphs.Sw_invSb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_NC1_1.png")

    plt.figure(4)
    plt.semilogy(cur_epochs, train_graphs.Sw_dSb)
    plt.semilogy(cur_epochs, test_graphs.Sw_dSb)
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Sw/Sb')
    plt.title('NC1: Activation Collapse')
    _y = train_graphs.Sw_dSb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = test_graphs.Sw_dSb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_NC1_2.png")

    plt.figure(5)
    plt.semilogy(cur_epochs, train_graphs.Sw)
    plt.semilogy(cur_epochs, test_graphs.Sw)
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Sw')
    plt.title('Within-class covariance')
    _y = train_graphs.Sw[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = test_graphs.Sw[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_Sw.png")

    plt.figure(6)
    plt.semilogy(cur_epochs, train_graphs.Sb)
    plt.semilogy(cur_epochs, test_graphs.Sb)
    plt.legend(["Train", "Test"])
    plt.xlabel('Epoch')
    plt.ylabel('Sb')
    plt.title('Between-class covariance')
    _y = train_graphs.Sb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    _y = test_graphs.Sb[-1]
    plt.text(_x, _y, str(round(_y, rd)))
    plt.savefig(f"{no}_Sb.png")

    if loss_name == "MSELoss":
        plt.figure(7)
        plt.plot(cur_epochs, train_graphs.LNC1)
        plt.plot(cur_epochs, test_graphs.LNC1)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('LNC1')
        plt.title('LNC1')
        _y = train_graphs.LNC1[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = test_graphs.LNC1[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        plt.savefig(f"{no}_LNC1.png")

    if other_NC:
        plt.figure(8)
        plt.plot(cur_epochs, train_graphs.norm_M_CoV)
        plt.plot(cur_epochs, test_graphs.norm_M_CoV)
        plt.plot(cur_epochs, train_graphs.norm_W_CoV)
        plt.legend(['Train Class Means','Test Class Means', 'Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Std/Avg of Norms')
        plt.title('NC2: Equinorm')
        _y = train_graphs.norm_M_CoV[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = test_graphs.norm_M_CoV[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = train_graphs.norm_W_CoV[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        plt.savefig(f"{no}_NC2_equinorm.png")

        plt.figure(9)
        plt.plot(cur_epochs, train_graphs.cos_M)
        plt.plot(cur_epochs, test_graphs.cos_M)
        plt.plot(cur_epochs, train_graphs.cos_W)
        plt.legend(['Class Means', 'Test Class Means', 'Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Avg|Cos + 1/(C-1)|')
        plt.title('NC2: Maximal Equiangularity')
        _y = train_graphs.cos_M[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = test_graphs.cos_M[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = train_graphs.cos_W[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        plt.savefig(f"{no}_NC2_equiangularity.png")

        plt.figure(10)
        plt.plot(cur_epochs, train_graphs.W_M_dist)
        plt.plot(cur_epochs, test_graphs.W_M_dist)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('||W^T - H||^2')
        plt.title('NC3: Self Duality')
        _y = train_graphs.W_M_dist[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = test_graphs.W_M_dist[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        plt.savefig(f'{no}_NC3_Self Duality.png')

        plt.figure(11)
        plt.plot(cur_epochs, train_graphs.NCC_mismatch)
        plt.plot(cur_epochs, test_graphs.NCC_mismatch)
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Proportion Mismatch from NCC')
        plt.title('NC4: Convergence to NCC')
        _y = train_graphs.NCC_mismatch[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        _y = test_graphs.NCC_mismatch[-1]
        plt.text(_x, _y, str(round(_y, rd)))
        plt.savefig(f'{no}_NC4_Convergence to NCC.png')

    plt.show()
    #print(f"Epoch: {epoch} Plotting finished")