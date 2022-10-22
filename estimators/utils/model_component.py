import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy

# spatialGCN
class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=0.0):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta0 = nn.Linear(in_channels, in_channels, bias=False)
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class SaGcnGru(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels,hidden_size, output_size, dropout=.0,):
        super(SaGcnGru,self).__init__()
        self.sa_gcn = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.sa_gcn_2 = spatialGCN(sym_norm_Adj_matrix,out_channels,out_channels,dropout)
        self.gru = nn.GRU(input_size=out_channels,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        sa_gcn_out = self.sa_gcn(x) #(b,n,t,f_out)
        # sa_gcn_out = self.sa_gcn_2(sa_gcn_out)
        # sa_gcn_out = F.relu(sa_gcn_out)
        batch_size, num_of_vertices, num_of_timesteps, out_channels = sa_gcn_out.shape
        sa_gcn_out = sa_gcn_out.reshape((-1,num_of_timesteps,out_channels))
        gru_out, gru_hidden = self.gru(sa_gcn_out)
        gru_hidden = gru_hidden.squeeze()
        # print('gru hidden',gru_hidden.squeeze().shape)
        last_gru_out = gru_out[:,-1,:].squeeze()
        # print('last_gru_out',last_gru_out.shape)
        # last_gru_out = gru_hidden
        # print(last_gru_out.shape)
        output_1 = self.fc1(last_gru_out)
        output_2 = self.fc2(last_gru_out)
        return output_1, output_2


class gcn_gru_unit(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels,hidden_size,dropout=.0,):
        super(gcn_gru_unit,self).__init__()
        self.sa_gcn = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.gru = nn.GRU(input_size=out_channels,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.hidden_size = hidden_size
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        sa_gcn_out = self.sa_gcn(x) #(b,n,t,f_out)
        batch_size, num_of_vertices, num_of_timesteps, out_channels = sa_gcn_out.shape
        sa_gcn_out = sa_gcn_out.reshape((-1,num_of_timesteps,out_channels))
        gru_out, gru_hidden = self.gru(sa_gcn_out)
        gru_hidden = gru_hidden.squeeze()
        last_gru_out = gru_out[:,-1,:].squeeze() #(b*n, hidden_size)
        last_gru_out = last_gru_out.reshape(batch_size, num_of_vertices,self.hidden_size) # batch_size,num_of_vertices,hidden_size
        return last_gru_out #(b,n,hidden_size)

class gru_gcn_unit(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0):
        super(gru_gcn_unit, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.sa_gcn = spatialGCN(sym_norm_Adj_matrix, hidden_size, out_channels, dropout)
        self.hidden_size = hidden_size
    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        gru_out = gru_out.reshape(batch_size, num_of_vertices, num_of_timesteps,self.hidden_size)
        sa_gcn_out = self.sa_gcn(gru_out) #(b,n,t,f_out)
        sa_gcn_out = sa_gcn_out[:,:,-1,:].squeeze()
        return sa_gcn_out #(b,n,f_out)


class gru_unit(nn.Module):
    def __init__(self,in_channels,hidden_size):
        super(gru_unit, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(batch_size, num_of_vertices,self.hidden_size) # batch_size,num_of_vertices,hidden_size
        return last_gru_out #(b,n,hidden_size)

class Attention(nn.Module):
    def __init__(self,att_input_size, att_out_size):
        super(Attention, self).__init__()
        self.k_weight = nn.Linear(att_input_size,att_out_size)
        self.q_weight = nn.Linear(att_input_size,att_out_size)
        self.v_weight = nn.Linear(att_input_size,att_out_size)

    def forward(self,x):
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = x
        b, n, f_out = gru_gcn_out.shape
        b, n, hidden_size = gcn_gru_out.shape
        gru_gcn_out = gru_gcn_out.reshape(-1,f_out)
        gru_gcn_out = torch.unsqueeze(gru_gcn_out,1)

        gcn_out = gcn_out.reshape(-1,f_out)
        gcn_out = torch.unsqueeze(gcn_out,1)

        gcn_gru_out = gcn_gru_out.reshape(-1,hidden_size)
        gcn_gru_out = torch.unsqueeze(gcn_gru_out,1)

        gru_out = gru_out.reshape(-1,hidden_size)
        gru_out = torch.unsqueeze(gru_out,1)

        att_input = torch.cat([gru_gcn_out,gcn_gru_out,gcn_out,gru_out],dim=1)

        k = self.k_weight(att_input)
        q = self.q_weight(att_input)
        v = self.v_weight(att_input)
        s = F.softmax(torch.matmul(k,q.transpose(1,2)), dim=-1)
        att_out = torch.matmul(s,v) #(b*n, 4, hidden_size)
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = att_out[:,0,:], att_out[:,1,:], att_out[:,2,:], att_out[:,3,:]
        # print('gru_gcn_out fin',gru_gcn_out.shape)
        return gru_gcn_out, gcn_gru_out, gru_out, gcn_out

class AttentionComponent(nn.Module):
    def __init__(self,att_input_size, att_out_size):
        super(AttentionComponent, self).__init__()
        self.k_weight = nn.Linear(att_input_size,att_out_size)
        self.q_weight = nn.Linear(att_input_size,att_out_size)
        self.v_weight = nn.Linear(att_input_size,att_out_size)

    def forward(self,x):
        gru_gcn_out, gcn_gru_out = x
        b, n, f_out = gru_gcn_out.shape
        # b, n, f_out = gcn_out.shape
        b, n, hidden_size = gcn_gru_out.shape

        gru_gcn_out = gru_gcn_out.reshape(-1,f_out)
        gru_gcn_out = torch.unsqueeze(gru_gcn_out,1)

        # gcn_out = gcn_out.reshape(-1,f_out)
        # gcn_out = torch.unsqueeze(gcn_out,1)

        gcn_gru_out = gcn_gru_out.reshape(-1,hidden_size)
        gcn_gru_out = torch.unsqueeze(gcn_gru_out,1)

        # gru_out = gru_out.reshape(-1,hidden_size)
        # gru_out = torch.unsqueeze(gru_out,1)

        att_input = torch.cat([gru_gcn_out,gcn_gru_out],dim=1)

        k = self.k_weight(att_input)
        q = self.q_weight(att_input)
        v = self.v_weight(att_input)
        s = F.softmax(torch.matmul(k,q.transpose(1,2)), dim=-1)
        att_out = torch.matmul(s,v) #(b*n, 4, hidden_size)
        gru_gcn_out, gcn_gru_out = att_out[:,0,:], att_out[:,1,:]
        # print('gru_gcn_out fin',gru_gcn_out.shape)
        return gru_gcn_out, gcn_gru_out

class AttentionComponentC(nn.Module):
    def __init__(self,att_input_size, att_out_size):
        super(AttentionComponentC, self).__init__()
        self.k_weight = nn.Linear(att_input_size,att_out_size)
        self.q_weight = nn.Linear(att_input_size,att_out_size)
        self.v_weight = nn.Linear(att_input_size,att_out_size)

    def forward(self,x):
        gcn_gru_out, gru_out, gcn_out = x
        b, n, f_out = gcn_out.shape
        # b, n, f_out = gcn_out.shape
        b, n, hidden_size = gcn_gru_out.shape


        gcn_out = gcn_out.reshape(-1,f_out)
        gcn_out = torch.unsqueeze(gcn_out,1)

        gcn_gru_out = gcn_gru_out.reshape(-1,hidden_size)
        gcn_gru_out = torch.unsqueeze(gcn_gru_out,1)

        gru_out = gru_out.reshape(-1,hidden_size)
        gru_out = torch.unsqueeze(gru_out,1)

        att_input = torch.cat([gcn_gru_out, gru_out, gcn_out],dim=1)

        k = self.k_weight(att_input)
        q = self.q_weight(att_input)
        v = self.v_weight(att_input)
        s = F.softmax(torch.matmul(k,q.transpose(1,2)), dim=-1)
        att_out = torch.matmul(s,v) #(b*n, 4, hidden_size)
        gcn_gru_out, gru_out, gcn_out = att_out[:,0,:], att_out[:,1,:], att_out[:,2,:]
        # print('gru_gcn_out fin',gru_gcn_out.shape)
        return gcn_gru_out, gru_out, gcn_out

class AttentionFake(nn.Module):
    def __init__(self,att_input_size, att_out_size):
        super(AttentionFake, self).__init__()
        self.k_weight = nn.Linear(att_input_size,att_out_size)
        self.q_weight = nn.Linear(att_input_size,att_out_size)
        self.v_weight = nn.Linear(att_input_size,att_out_size)
    def forward(self,x):
        gru_gcn_out = x
        b, n, f_out = gru_gcn_out.shape
        att_input = torch.cat([gru_gcn_out],dim=1)
        k = self.k_weight(att_input)
        q = self.q_weight(att_input)
        v = self.v_weight(att_input)
        s = F.softmax(torch.matmul(k,q.transpose(1,2)), dim=-1)
        att_out = torch.matmul(s,v) #(b*n, 4, hidden_size)
        gru_gcn_out = att_out[:,0,:], att_out[:,1,:], att_out[:,2,:], att_out[:,3,:]
        # print('gru_gcn_out fin',gru_gcn_out.shape)
        return att_out

class ParalleStGcn(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcn, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 2*(hidden_size+out_channels)
        # self.concat_size = hidden_size
        self.att = Attention(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)


    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_gcn_out_unsq = torch.unsqueeze(gru_gcn_out,dim=2)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = self.att((gru_gcn_out, gcn_gru_out, gru_out, gcn_out))


        # concat
        concat_out = torch.cat([gru_gcn_out,gcn_gru_out,gru_out,gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2

class ParalleStGcnSingle(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcn, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 2*(hidden_size+out_channels)
        self.att = Attention(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)


    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_gcn_out_unsq = torch.unsqueeze(gru_gcn_out,dim=2)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = self.att((gru_gcn_out, gcn_gru_out, gru_out, gcn_out))


        # concat
        concat_out = torch.cat([gru_gcn_out,gcn_gru_out,gru_out,gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1, output_2


class ParalleStGcnComponentC(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnComponentC, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 3*hidden_size
        # self.concat_size = hidden_size
        self.att = AttentionComponentC(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)
        # attention
        gcn_gru_out, gru_out, gcn_out = self.att((gcn_gru_out, gru_out, gcn_out))

        # concat
        concat_out = torch.cat([gcn_gru_out, gru_out, gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2

class ParalleStGcnS(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnS, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = hidden_size*3
        self.att = AttentionComponentC(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)

        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gcn_gru_out, gcn_out = self.att((gru_gcn_out,gcn_gru_out, gcn_out))
        # concat
        concat_out = torch.cat([gru_gcn_out, gcn_gru_out, gcn_out],dim=-1)
        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnT(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnT, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.concat_size = 2*(hidden_size+out_channels)
        self.concat_size = hidden_size*3
        self.att = AttentionComponentC(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)


        # attention
        gru_gcn_out, gcn_gru_out, gru_out = self.att((gru_gcn_out,gcn_gru_out, gru_out))
        # concat
        concat_out = torch.cat([gru_gcn_out, gcn_gru_out, gru_out],dim=-1)
        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnTS(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnTS, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = hidden_size*3
        self.att = AttentionComponentC(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gru_out, gcn_out = self.att((gru_gcn_out,gru_out, gcn_out))
        # concat
        concat_out = torch.cat([gru_gcn_out, gru_out, gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2

class ParalleStGcnST(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnST, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = hidden_size*3
        self.att = AttentionComponentC(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gcn_gru_out, gru_out, gcn_out = self.att((gcn_gru_out,gru_out, gcn_out))
        # concat
        concat_out = torch.cat([gcn_gru_out, gru_out, gcn_out],dim=-1)
        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnATT(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnATT, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 2*(hidden_size+out_channels)
        # self.concat_size = hidden_size
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)


    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_gcn_out_unsq = torch.unsqueeze(gru_gcn_out,dim=2)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)
        # concat
        concat_out = torch.cat([gru_gcn_out,gcn_gru_out,gru_out,gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnFake(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnFake, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = spatialGCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = hidden_size
        self.att = AttentionFake(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)

        # attention
        gru_gcn_out = self.att(gru_gcn_out)
        # concat
        concat_out = gru_gcn_out
        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2

class Gru(nn.Module):

    def __init__(self,in_channels,hidden_size,output_size):
        super(Gru, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(-1,self.hidden_size)
        last_gru_out_1 = self.fc1(last_gru_out)
        return last_gru_out_1


class GruMulti(nn.Module):

    def __init__(self,in_channels,hidden_size,output_size):
        super(GruMulti, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(-1,self.hidden_size)
        last_gru_out_1 = self.fc1(last_gru_out)
        last_gru_out_2 = self.fc2(last_gru_out)
        return last_gru_out_1, last_gru_out_2



def slice_by_time(pop_mean_df, pop_var_df, interval=5):

    m_list = []
    v_list = []
    for i in range(interval):
        pop_mean_slice = pop_mean_df.iloc[i::interval,:]
        m_list.append(pop_mean_slice)
        pop_var_slice = pop_var_df.iloc[i::interval,:]
        v_list.append(pop_var_slice)
    return m_list, v_list

def extract_feat_lable_by_day(seq_len, pre_len, pop_mean_one_day,pop_variance_one_day):
    pop_mean_one_day = pop_mean_one_day.values
    pop_variance_one_day = pop_variance_one_day.values
    features = []
    labels = []
    for i in range(len(pop_mean_one_day) - seq_len - pre_len):
        pop_mean_one_sample = pop_mean_one_day[i:i + seq_len]
        pop_mean_one_sample = np.expand_dims(pop_mean_one_sample, axis=2)
        pop_variance_one_sample = pop_variance_one_day[i:i + seq_len]
        pop_variance_one_sample = np.expand_dims(pop_variance_one_sample, axis=2)
        one_sample_feature = np.concatenate([pop_mean_one_sample, pop_variance_one_sample], axis=2)
        pop_mean_one_label = pop_mean_one_day[i + seq_len:i + seq_len + pre_len]
        pop_mean_one_label = np.expand_dims(pop_mean_one_label, axis=2)
        pop_variance_one_label = pop_variance_one_day[i + seq_len:i + seq_len + pre_len]
        pop_variance_one_label = np.expand_dims(pop_variance_one_label, axis=2)
        one_sample_label = np.concatenate([pop_mean_one_label, pop_variance_one_label], axis=2)
        features.append(one_sample_feature)
        labels.append(one_sample_label)
    features = np.array(features)
    print('features', features.shape)
    labels = np.array(labels)
    print('labels', labels.shape)
    return features, labels

def extract_feat_label_by_day_with_interval(seq_len, pre_len, Mean, Var, interval):
    m_list, v_list = slice_by_time(Mean, Var, interval=interval)
    feat_five_mins, label_five_mins = zip(*[extract_feat_lable_by_day(seq_len, pre_len, m, v) for m, v in list(zip(m_list, v_list))])
    feat_five_mins = np.concatenate(feat_five_mins, axis=0)
    label_five_mins = np.concatenate(label_five_mins, axis=0)
    return feat_five_mins, label_five_mins

def train_test_valid_split(features, labels, train_rate, validation_rate):
    features = np.array(features)
    labels = np.array(labels)
    idx = list(range(len(features)))
    idx_ordered = copy.copy(idx)
    idx_train = idx[:int(len(idx) * train_rate)]
    idx_validation = idx_ordered[int(len(idx)*train_rate):int(len(idx)*(train_rate+validation_rate))]
    idx_test = idx[int(len(idx) * (train_rate + validation_rate)):]
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_validation)
    print('len(idx_validation)', len(idx_validation))
    np.random.shuffle(idx_test)
    print('len(idx_test)', len(idx_test))
    return features, labels, idx_train, idx_validation, idx_test

def eval_dist(out1, out2, label1, label2):
    delta_x = (out1 - label1)**2
    delta_y = (out2 - label2)**2
    delta_xy = delta_x + delta_y
    res = np.average(np.sqrt(delta_xy))
    return res

def eval_kl(u_q, v_q, u_p, v_p):
    """
    Parameters
    ----------
    P(x) ~ N(u_p, v_p) true data
    Q(x) ~ N(u_q, v_q) model output
    KL = P(x)Log(P(x)/Q(x))
    out1 : u_q
    out2 : v_q
    label1: u_p
    label2: v_p

    kl = 1/2(log(v_q**2) - log(v_p**2)) + (v_p + (u_p - u_q)**2)/(2*v_q)) - 1/2
    Returns
    -------
    """
    v_q_i = set(list(np.where(v_q>0)[0]))
    v_p_i = set(list(np.where(v_p>0)[0]))
    v_i = list(v_p_i.intersection(v_q_i))

    u_q = u_q[v_i]
    v_q = v_q[v_i]
    u_p = u_p[v_i]
    v_p = v_p[v_i]

    kl =  0.5*(np.log(v_q**2) - np.log(v_p**2)) + (v_p + (u_p - u_q)**2)/(2*v_q) - 0.5
    kl_avg = np.average(kl)
    return kl_avg


def integrated_loss(out1, out2, label1,label2):
    loss1 = F.mse_loss(out1, label1)
    loss2 = F.mse_loss(out2, label2)
    loss = loss1 + loss2
    return loss

