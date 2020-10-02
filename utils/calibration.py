import torch
from utils.draw_picture import draw_loss, draw_two

def get_acc(y_pred, y):
    acc_1 = 0
    a_count = 0
    for i in range(len(y)):
        if torch.argmax(y[i]) == torch.argmax(y_pred[i]):
            acc_1 += 1

    return acc_1 / len(y)

def calibrate(y_pred, y_pred2, y, val_pred1, val_pred2, y_val, scale, bins, la):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    y1_max = y_pred.max(1, keepdim=True)
    y_pred_sort = torch.sort(y1_max[0], 0)
    y2_max = y_pred2.max(1, keepdim=True)
    y_pred_sort2 = torch.sort(y2_max[0], 0)
    val1_max = val_pred1.max(1, keepdim=True)
    val_pred_sort = torch.sort(val1_max[0], 0)
    val2_max = val_pred2.max(1, keepdim=True)
    val_pred_sort2 = torch.sort(val2_max[0], 0)
    y_pred_c = torch.tensor([[0.0]*10]*400)
    y_pred_c = y_pred_c.to(device)
    step = int(400*scale/bins)
    for i in range(bins):
        val1, val2, val_y, val_y2, val_index, val_index2, y_t, pred_t, pred_t2, pred_t_c = [], [], [], [], [], [], [], [], [], []
        for j in range(len(val1_max[0])):
            if val1_max[0][j]>=val_pred_sort[0][i*step] and val1_max[0][j]<=val_pred_sort[0][(i+1)*step-1]:
                val_index.append(j)
        for j in val_index:
            val1.append(val_pred1[j])
            val_y.append(y_val[j])
        for j in range(len(val2_max[0])):
            if val2_max[0][j]>=val_pred_sort2[0][i*step] and val2_max[0][j]<=val_pred_sort2[0][(i+1)*step-1]:
                val_index2.append(j)
        for j in val_index2:
            val2.append(val_pred2[j])
            val_y2.append(y_val[j])
        acc1 = get_acc(val1, val_y)
        acc2 = get_acc(val2, val_y2)
        ability1 = 1-10*(1-acc1)/9+la
        ability2 = 1-10*(1-acc2)/9+la
        print(ability1, ability2)

        y1_con = torch.sum(y_pred_sort[0][i*step:(i+1)*step])
        y2_con = torch.sum(y_pred_sort2[0][i*step:(i+1)*step])
        y_start = y_pred_sort[0][i*step]
        y_end = y_pred_sort[0][(i+1)*step-1]
        ca = (ability1*y2_con/(ability2*y1_con))

        index, y_pred_max, y_pred_max_c, pred_t_np, pred_t_c_np = [], [], [], [], []
        for j in range(len(y1_max[0])):
            if y1_max[0][j]>=y_start and y1_max[0][j]<=y_end:
                index.append(j)
        for j in index:
            pred_t.append(y_pred[j])
            pred_t2.append(y_pred2[j])
            y_t.append(y[j])
            y_pred_max.append(torch.max(y_pred[j], y_pred2[j]))
        print(get_acc(pred_t, y_t), get_acc(pred_t2, y_t), get_acc(y_pred_max, y_t))
        for j in index:
            y_pred_c[j] = y_pred[j] * ca
            y_pred_c[j] = y_pred_c[j].to(device)
        for j in index:
            pred_t_c.append(y_pred_c[j])
            y_pred_max_c.append(torch.max(y_pred_c[j], y_pred2[j]))
        print(get_acc(pred_t_c, y_t), get_acc(pred_t2, y_t), get_acc(y_pred_max_c, y_t))


    return y_pred_c
