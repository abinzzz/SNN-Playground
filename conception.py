import torch

v = torch.rand([8])
v_th = 0.5
spike = (v >= v_th).to(v)

#通过下面对比发现，超过阈值的会被转化为1，没有超过阈值的会被转化为0
print('v=',v)
print('spike =', spike)



## 步进模式：单步与多步
import torch
from spikingjelly.activation_based import neuron

# net = neuron.IFNode(step_mode='m')
# # 'm' is the multi-step mode
# net.step_mode = 's'
# # 's' is the single-step mode


#单步模式
net_s = neuron.IFNode(step_mode='s')
T = 3
N = 1
C = 3
H = 8
W = 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = []
#print("x_seq=",x_seq)
for t in range(T):
    x = x_seq[t]  # x.shape = [N, C, H, W]
    y = net_s(x)  # y.shape = [N, C, H, W]
    #print("x=",x)
    #print("y=",y)
    #print("T=",T)
    y_seq.append(y.unsqueeze(0))
    
#y_seq是有三个元素[1,1,3,8,8]的列表
y_seq_s = torch.cat(y_seq) #[3,1,3,8,8]
#print("y_seq(单步)=",y_seq_s)
# y_seq.shape = [T, N, C, H, W]



# 与多步模式做对比
net_m = neuron.IFNode(step_mode='m')
# T = 4
# N = 1
# C = 3
# H = 8
# W = 8
#x_seq = torch.rand([T, N, C, H, W])
y_seq_m = net_m(x_seq)
#print("y_seq(多步)=",y_seq)
is_equal = torch.equal(y_seq_s, y_seq_m)
#print("输出是否相同:", is_equal)



## 状态保存与重置
from spikingjelly.activation_based import neuron

net_s = neuron.IFNode(step_mode='s')
x = torch.rand([4])
print(net_s)
print(f'the initial v={net_s.v}')
y = net_s(x)
print(f'x={x}')
print(f'y={y}')
print(f'v={net_s.v}')